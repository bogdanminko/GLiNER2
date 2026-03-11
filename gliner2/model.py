"""
GLiNER2 Extractor Model with Optimized Batch Processing

This module contains the core Extractor model that accepts PreprocessedBatch
directly for efficient GPU-only forward passes.

Two encoder modes are supported:

- **Uni-encoder** (default): single transformer for text + schema.
- **Bi-encoder**: separate text and schema transformers, enabling schema
  embedding caching and heterogeneous encoder pairs.

See ``ExtractorConfig`` for all configuration options.
"""

import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gliner.modeling.span_rep import SpanRepLayer
from gliner2.layers import CountLSTMoE, CountLSTM, create_mlp, CountLSTMv2
from gliner2.processor import SchemaTransformer, PreprocessedBatch, SamplingConfig
from safetensors.torch import save_file, load_file
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
)


class CrossFuserLayer(nn.Module):
    """Single bidirectional cross-attention layer: W,E → W',E'."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # text attends to schema
        self.text_cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_size)

        # schema attends to text
        self.schema_cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.schema_norm = nn.LayerNorm(hidden_size)

        # Zero-init output projections so CrossFuser starts as identity
        nn.init.zeros_(self.text_cross_attn.out_proj.weight)
        nn.init.zeros_(self.text_cross_attn.out_proj.bias)
        nn.init.zeros_(self.schema_cross_attn.out_proj.weight)
        nn.init.zeros_(self.schema_cross_attn.out_proj.bias)

    def forward(
        self,
        W: torch.Tensor,          # (B, L, D)
        E: torch.Tensor,          # (B, C, D)
        text_mask: Optional[torch.Tensor] = None,    # (B, L)
        schema_mask: Optional[torch.Tensor] = None,  # (B, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # text ← schema:  Q=W, K=V=E
        text_key_padding = ~schema_mask.bool() if schema_mask is not None else None
        t_out, _ = self.text_cross_attn(W, E, E, key_padding_mask=text_key_padding)
        W = self.text_norm(W + t_out)

        # schema ← text:  Q=E, K=V=W
        schema_key_padding = ~text_mask.bool() if text_mask is not None else None
        s_out, _ = self.schema_cross_attn(E, W, W, key_padding_mask=schema_key_padding)
        E = self.schema_norm(E + s_out)

        return W, E


class CrossFuser(nn.Module):
    """Stack of bidirectional cross-attention layers."""

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossFuserLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, W, E, text_mask=None, schema_mask=None):
        for layer in self.layers:
            W, E = layer(W, E, text_mask, schema_mask)
        return W, E


class ExtractorConfig(PretrainedConfig):
    """
    Configuration for the Extractor model.

    Attributes:
        model_name: HF model id or path for the text encoder.
        max_width: Maximum span width (in tokens) considered during extraction.
        counting_layer: Count embedding variant (``"count_lstm"``, ``"count_lstm_moe"``, ``"count_lstm_v2"``).
        token_pooling: Sub-word pooling strategy (``"first"``).
        max_len: Optional hard limit on input word-tokens (``None`` = no limit).
        encoder_mode: ``"uni"`` (default) — single encoder for text+schema;
            ``"bi"`` — separate text and schema encoders.
        schema_model_name: HF model id for the schema encoder (bi-encoder only).
            Defaults to ``model_name`` when ``None``.
        schema_projection_dim: If set, both encoders are projected to this
            dimensionality (required when their hidden sizes differ).
    """
    model_type = "extractor"

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            max_width: int = 8,
            counting_layer: str = "count_lstm",
            token_pooling: str = "first",
            max_len: int = None,
            # Bi-encoder settings
            encoder_mode: str = "uni",
            schema_model_name: str = None,
            schema_projection_dim: int = None,
            cross_fuser_heads: int = 0,
            cross_fuser_layers: int = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_width = max_width
        self.counting_layer = counting_layer
        self.token_pooling = token_pooling
        self.max_len = max_len
        self.encoder_mode = encoder_mode
        self.schema_model_name = schema_model_name
        self.schema_projection_dim = schema_projection_dim
        self.cross_fuser_heads = cross_fuser_heads
        self.cross_fuser_layers = cross_fuser_layers


class Extractor(PreTrainedModel):
    """
    GLiNER2 Extractor Model.

    Supports two encoder architectures (set via ``ExtractorConfig.encoder_mode``):

    - **Uni-encoder** (``"uni"``, default): a single transformer encodes the
      concatenated text + schema sequence.
    - **Bi-encoder** (``"bi"``): text and schema are encoded by independent
      transformers (potentially different models / tokenizers). When
      ``schema_model_name`` is provided in the config, a separate schema
      encoder is loaded; otherwise the same architecture is used for both.
      A projection layer is added automatically when the two encoders have
      different hidden sizes (controlled by ``schema_projection_dim``).

    This model accepts PreprocessedBatch for efficient training.
    Use processor.collate_fn_train() to create batches.

    Example:
        >>> processor = SchemaTransformer(model_name)
        >>> model = Extractor.from_pretrained(repo_id)
        >>>
        >>> # Training
        >>> loader = DataLoader(dataset, collate_fn=processor.collate_fn_train)
        >>> for batch in loader:
        ...     batch = batch.to(device)
        ...     loss = model(batch)["total_loss"]
    """
    config_class = ExtractorConfig

    def __init__(self, config: ExtractorConfig, encoder_config=None, schema_encoder_config=None,
                 tokenizer=None, schema_tokenizer=None):
        super().__init__(config)
        self.config = config
        self.max_width = config.max_width

        # Resolve schema tokenizer for bi-encoder
        _schema_tok = schema_tokenizer
        if config.encoder_mode == "bi" and _schema_tok is None and tokenizer is None:
            schema_name = config.schema_model_name or config.model_name
            if schema_name != config.model_name:
                _schema_tok = AutoTokenizer.from_pretrained(schema_name)

        # Initialize processor
        if tokenizer is not None:
            self.processor = SchemaTransformer(
                tokenizer=tokenizer,
                schema_tokenizer=_schema_tok,
                token_pooling=config.token_pooling,
                encoder_mode=config.encoder_mode,
            )
        else:
            self.processor = SchemaTransformer(
                config.model_name,
                schema_tokenizer=_schema_tok,
                token_pooling=config.token_pooling,
                encoder_mode=config.encoder_mode,
            )

        # Resolve attention implementation from env
        _flash = os.environ.get("FLASH_ATTN", "").lower() in ("1", "true", "yes")
        _attn_impl = "flash_attention_2" if _flash else None
        if _flash:
            _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            _dtype = None

        # Load encoder
        if encoder_config is not None:
            self.encoder = AutoModel.from_config(encoder_config, trust_remote_code=True)
        else:
            self.encoder = AutoModel.from_pretrained(
                config.model_name, trust_remote_code=True,
                attn_implementation=_attn_impl,
                dtype=_dtype,
            )

        _actual = getattr(self.encoder.config, "_attn_implementation", None)
        print(f"[GLiNER2] Attention: requested={_attn_impl}, actual={_actual}")

        self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.hidden_size = self.encoder.config.hidden_size

        # Bi-encoder: initialize schema encoder
        if config.encoder_mode == "bi":
            if config.schema_model_name is not None:
                # Separate schema encoder (different model)
                if schema_encoder_config is not None:
                    self.schema_encoder = AutoModel.from_config(schema_encoder_config, trust_remote_code=True)
                else:
                    self.schema_encoder = AutoModel.from_pretrained(
                        config.schema_model_name, trust_remote_code=True,
                        attn_implementation=_attn_impl,
                        dtype=_dtype,
                    )
                schema_tok = self.processor.schema_tokenizer or self.processor.tokenizer
                self.schema_encoder.resize_token_embeddings(len(schema_tok))
                self._shared_encoder = False
            else:
                # Shared encoder (default): same model for text and schema
                self.schema_encoder = self.encoder
                self._shared_encoder = True

            schema_hidden = self.schema_encoder.config.hidden_size

            if config.schema_projection_dim is not None:
                self.text_proj = nn.Linear(self.hidden_size, config.schema_projection_dim)
                self.schema_proj = nn.Linear(schema_hidden, config.schema_projection_dim)
                self.hidden_size = config.schema_projection_dim
            elif schema_hidden != self.hidden_size:
                raise ValueError(
                    f"Text encoder hidden_size={self.hidden_size} != schema encoder "
                    f"hidden_size={schema_hidden}. Set schema_projection_dim to align them."
                )
            else:
                self.text_proj = None
                self.schema_proj = None
        else:
            self.schema_encoder = None
            self.text_proj = None
            self.schema_proj = None
            self._shared_encoder = False

        # Cross-attention fuser (bi-encoder only)
        if config.cross_fuser_heads > 0 and config.encoder_mode == "bi":
            self.cross_fuser = CrossFuser(
                hidden_size=self.hidden_size,
                num_heads=config.cross_fuser_heads,
                num_layers=config.cross_fuser_layers,
            )
        else:
            self.cross_fuser = None

        # Span representation layer
        self.span_rep = SpanRepLayer(
            span_mode="markerV0",
            hidden_size=self.hidden_size,
            max_width=self.max_width,
            dropout=0.1,
        )

        # Classifier for classification tasks
        self.classifier = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=1,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

        # Count prediction layer
        self.count_pred = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=20,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

        # Count embedding module
        if config.counting_layer == "count_lstm":
            self.count_embed = CountLSTM(self.hidden_size)
        elif config.counting_layer == "count_lstm_moe":
            self.count_embed = CountLSTMoE(
                hidden_size=self.hidden_size,
                n_experts=4,
                ffn_mult=2,
                dropout=0.1
            )
        elif config.counting_layer == "count_lstm_v2":
            self.count_embed = CountLSTMv2(hidden_size=self.hidden_size)

        # LoRA adapter state
        self._lora_layers = {}
        self._adapter_config = None

        self._print_config(config)

    def _print_config(self, config):
        print("=" * 60)
        print("Model Configuration")
        print("=" * 60)
        print(f"Encoder mode       : {config.encoder_mode}")
        print(f"Text encoder       : {config.model_name}")
        if config.encoder_mode == "bi":
            print(f"Schema encoder     : {config.schema_model_name or config.model_name}")
            if config.schema_projection_dim:
                print(f"Projection dim     : {config.schema_projection_dim}")
        print(f"Counting layer     : {config.counting_layer}")
        print(f"Token pooling      : {config.token_pooling}")
        print("=" * 60)

    # =========================================================================
    # Main Forward Pass
    # =========================================================================

    def forward(
            self,
            batch: PreprocessedBatch,
            return_individual_losses: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass on preprocessed batch.

        Args:
            batch: PreprocessedBatch from processor.collate_fn_train()
            return_individual_losses: If True, return per-sample losses

        Returns:
            Dict with:
                - total_loss: Sum of all losses
                - classification_loss: Classification task loss
                - structure_loss: Span extraction loss
                - count_loss: Count prediction loss
                - batch_size: Number of valid samples
        """
        if len(batch) == 0:
            return self._empty_loss_dict()

        device = next(self.parameters()).device
        batch = batch.to(device)

        # Encode batch through transformer
        all_token_embs, all_schema_embs = self._encode_batch(batch)

        # Compute losses for each sample
        cls_losses = []
        struct_losses = []
        count_losses = []
        individual = []
        valid_samples = 0

        for i in range(len(batch)):
            try:
                sample_losses = self._compute_sample_loss(
                    token_embeddings=all_token_embs[i],
                    embs_per_schema=all_schema_embs[i],
                    task_types=batch.task_types[i],
                    structure_labels=batch.structure_labels[i],
                    device=device
                )

                cls_losses.append(sample_losses["classification"])
                struct_losses.append(sample_losses["structure"])
                count_losses.append(sample_losses["count"])

                if return_individual_losses:
                    individual.append({
                        "total_loss": (
                                sample_losses["classification"] +
                                sample_losses["structure"] +
                                sample_losses["count"]
                        ).item(),
                        "classification_loss": sample_losses["classification"].item(),
                        "structure_loss": sample_losses["structure"].item(),
                        "count_loss": sample_losses["count"].item(),
                    })

                valid_samples += 1

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                zero = torch.tensor(0.0, device=device)
                cls_losses.append(zero)
                struct_losses.append(zero)
                count_losses.append(zero)

                if return_individual_losses:
                    individual.append({
                        "total_loss": 0.0,
                        "classification_loss": 0.0,
                        "structure_loss": 0.0,
                        "count_loss": 0.0,
                        "error": str(e)
                    })

        if valid_samples == 0:
            result = self._empty_loss_dict()
            if return_individual_losses:
                result["individual_losses"] = individual
            return result

        # Aggregate losses
        total_cls = torch.stack(cls_losses).sum()
        total_struct = torch.stack(struct_losses).sum()
        total_count = torch.stack(count_losses).sum()
        total_loss = total_cls + total_struct + total_count

        result = {
            "total_loss": total_loss,
            "classification_loss": total_cls,
            "structure_loss": total_struct,
            "count_loss": total_count,
            "batch_size": valid_samples
        }

        if return_individual_losses:
            result["individual_losses"] = individual

        return result

    def _empty_loss_dict(self) -> Dict[str, torch.Tensor]:
        """Return empty loss dictionary."""
        device = next(self.parameters()).device
        return {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "classification_loss": torch.tensor(0.0, device=device),
            "structure_loss": torch.tensor(0.0, device=device),
            "count_loss": torch.tensor(0.0, device=device),
            "batch_size": 0
        }

    # =========================================================================
    # Encoding
    # =========================================================================

    def _encode_batch(
            self,
            batch: PreprocessedBatch,
            cached_schema_embs: Optional[List[List[List[torch.Tensor]]]] = None,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Encode batch through transformer and extract embeddings.

        Args:
            batch: PreprocessedBatch with input_ids and attention_mask
            cached_schema_embs: Pre-computed schema embeddings (bi-encoder only).
                If provided, schema encoder is skipped.

        Returns:
            - all_token_embs: List of (text_len, hidden) per sample
            - all_schema_embs: List of schema embeddings per sample
        """
        if self.schema_encoder is not None:
            return self._encode_batch_bi(batch, cached_schema_embs)

        # Uni-encoder path (unchanged)
        outputs = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )
        token_embeddings = outputs.last_hidden_state

        return self.processor.extract_embeddings_from_batch(
            token_embeddings,
            batch.input_ids,
            batch
        )

    def _encode_batch_bi(
            self,
            batch: PreprocessedBatch,
            cached_schema_embs: Optional[List[List[List[torch.Tensor]]]] = None,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Bi-encoder: encode text and schema separately, with optional CrossFuser.

        Flow:
            TextEncoder(text) → W
            LabelEncoder(labels) → E  (or use cached)
            CrossFuser(W, E) → W', E'  (optional)
            extract(W', E') → word_embs, schema_embs

        Args:
            batch: PreprocessedBatch with bi-encoder fields populated
            cached_schema_embs: Pre-computed schema embeddings to skip schema encoding.
                When cross_fuser is None: List of extracted special-token embeddings.
                When cross_fuser is set: dict with 'tensor' and 'mask' keys.

        Returns:
            - all_token_embs: List of (text_len, hidden) per sample
            - all_schema_embs: List of schema embeddings per sample
        """
        # Encode text
        text_outputs = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        ).last_hidden_state

        if self.text_proj is not None:
            text_outputs = self.text_proj(text_outputs)

        # --- No CrossFuser: fast cached path (original behavior) ---
        if self.cross_fuser is None and cached_schema_embs is not None:
            all_token_embs = []
            for i in range(len(batch)):
                t_mappings = batch.mapped_indices[i]
                t_seq_len = batch.original_lengths[i]
                t_embs = text_outputs[i, :t_seq_len, :]

                word_embs = []
                bucket = []
                last_orig = None

                for j in range(t_seq_len):
                    _, orig_idx, _ = t_mappings[j]
                    emb = t_embs[j]
                    if last_orig is not None and orig_idx != last_orig and bucket:
                        word_embs.append(self.processor._aggregate(bucket))
                        bucket = []
                    bucket.append(emb)
                    last_orig = orig_idx

                if bucket:
                    word_embs.append(self.processor._aggregate(bucket))

                all_token_embs.append(
                    torch.stack(word_embs) if word_embs
                    else torch.empty(0, t_embs.shape[-1], device=t_embs.device)
                )

            return all_token_embs, cached_schema_embs

        # --- Get schema outputs (encode or use cached tensor) ---
        if cached_schema_embs is not None and self.cross_fuser is not None:
            # CrossFuser + cache: cached_schema_embs is a dict with tensor + mask
            schema_outputs = cached_schema_embs['tensor'].expand(text_outputs.shape[0], -1, -1)
            schema_mask = cached_schema_embs['mask'].expand(text_outputs.shape[0], -1)
        else:
            # Encode schema from scratch
            schema_outputs = self.schema_encoder(
                input_ids=batch.schema_input_ids,
                attention_mask=batch.schema_attention_mask
            ).last_hidden_state

            if self.schema_proj is not None:
                schema_outputs = self.schema_proj(schema_outputs)

            schema_mask = batch.schema_attention_mask

        # --- CrossFuser: W,E → W',E' ---
        if self.cross_fuser is not None:
            text_outputs, schema_outputs = self.cross_fuser(
                text_outputs, schema_outputs,
                text_mask=batch.attention_mask,
                schema_mask=schema_mask,
            )

        return self.processor.extract_embeddings_from_bi_encoder(
            text_outputs, schema_outputs, batch
        )

    # =========================================================================
    # Loss Computation
    # =========================================================================

    def _compute_sample_loss(
            self,
            token_embeddings: torch.Tensor,
            embs_per_schema: List[List[torch.Tensor]],
            task_types: List[str],
            structure_labels: List[Any],
            device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a single sample.

        Args:
            token_embeddings: (text_len, hidden) text token embeddings
            embs_per_schema: List of schema embeddings
            task_types: Task type for each schema
            structure_labels: Labels for each schema
            device: Computation device

        Returns:
            Dict with classification, structure, and count losses
        """
        cls_loss = torch.tensor(0.0, device=device)
        struct_loss = torch.tensor(0.0, device=device)
        count_loss = torch.tensor(0.0, device=device)

        # Compute span representations if needed
        has_span_task = any(t != "classifications" for t in task_types)
        span_info = None
        if has_span_task and token_embeddings.numel() > 0:
            span_info = self.compute_span_rep(token_embeddings)

        all_counts = []
        all_p_embs = []

        for i, task_type in enumerate(task_types):
            if not embs_per_schema[i]:
                continue

            schema_emb = torch.stack(embs_per_schema[i])

            if task_type == "classifications":
                # Classification loss
                cls_embeds = schema_emb[1:]  # Skip [P] token
                logits = self.classifier(cls_embeds).squeeze(-1)
                labels = torch.tensor(structure_labels[i], dtype=torch.float, device=device)
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
            else:
                # Structure loss
                structure = structure_labels[i]

                if structure[0] == 0:
                    # No instances to extract
                    continue

                if span_info is not None:
                    struct_loss = struct_loss + self.compute_struct_loss(
                        span_info["span_rep"],
                        schema_emb,
                        structure,
                        span_info["span_mask"]
                    )

                # Collect for count loss (skip entities)
                if task_type != "entities":
                    all_counts.append(min(structure[0], 19))
                    all_p_embs.append(schema_emb[0])

        # Count loss
        if all_counts and all_p_embs:
            counts = torch.tensor(all_counts, dtype=torch.long, device=device)
            p_embs = torch.stack(all_p_embs)
            count_loss = F.cross_entropy(self.count_pred(p_embs), counts, reduction="sum")

        return {
            "classification": cls_loss,
            "structure": struct_loss,
            "count": count_loss
        }

    # =========================================================================
    # Span Representation
    # =========================================================================

    def compute_span_rep(self, token_embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Compute span representations for token embeddings.

        Args:
            token_embeddings: (text_len, hidden) token embeddings

        Returns:
            Dict with span_rep, spans_idx, and span_mask
        """
        text_length = len(token_embeddings)
        device = token_embeddings.device

        spans_idx = []
        for i in range(text_length):
            for j in range(self.max_width):
                if i + j < text_length:
                    spans_idx.append((i, i + j))
                else:
                    spans_idx.append((-1, -1))

        spans_idx = torch.tensor([spans_idx], dtype=torch.long, device=device)

        # Mask invalid spans
        span_mask = (spans_idx[:, :, 0] == -1) | (spans_idx[:, :, 1] == -1)

        # Replace invalid with (0, 0) for safe indexing
        safe_spans = torch.where(
            span_mask.unsqueeze(-1),
            torch.zeros_like(spans_idx),
            spans_idx
        )

        # Compute span representations
        span_rep = self.span_rep(
            token_embeddings.unsqueeze(0),
            safe_spans
        ).squeeze(0)

        return {
            "span_rep": span_rep,
            "spans_idx": spans_idx,
            "span_mask": span_mask
        }

    def compute_struct_loss(
            self,
            span_rep: torch.Tensor,
            schema_emb: torch.Tensor,
            structure: List[Any],
            span_mask: torch.Tensor,
            masking_rate: float = 0.5
    ) -> torch.Tensor:
        """
        Compute structure extraction loss with negative span masking.

        Args:
            span_rep: (num_spans, hidden) span representations
            schema_emb: (num_fields + 1, hidden) schema embeddings
            structure: [count, spans] structure labels
            span_mask: (1, num_spans) mask for invalid spans
            masking_rate: Probability of masking negative spans

        Returns:
            Structure loss tensor
        """
        gold_count = min(structure[0], 19)
        struct_proj = self.count_embed(schema_emb[1:], gold_count)
        scores = torch.einsum('lkd,bpd->bplk', span_rep, struct_proj)

        # Create label tensor
        labs = torch.zeros_like(scores)

        for i in range(gold_count):
            gold_spans = structure[1][i]
            for k, span in enumerate(gold_spans):
                if span is None or span == (-1, -1):
                    continue
                if isinstance(span, tuple):
                    start, end = span
                    width = end - start
                    if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                        labs[i, k, start, width] = 1
                elif isinstance(span, list):
                    for sub in span:
                        if sub is None or sub == (-1, -1):
                            continue
                        start, end = sub
                        width = end - start
                        if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                            labs[i, k, start, width] = 1

        # Apply negative masking
        if masking_rate > 0.0 and self.training:
            negative = (labs == 0)
            random_mask = torch.rand_like(scores) < masking_rate
            to_mask = negative & random_mask
            loss_mask = (~to_mask).float()
        else:
            loss_mask = torch.ones_like(scores)

        # Compute masked loss
        loss = F.binary_cross_entropy_with_logits(scores, labs, reduction="none")
        loss = loss * loss_mask
        loss = loss.view(loss.shape[0], loss.shape[1], -1) * (~span_mask[0]).float()

        return loss.sum()

    # =========================================================================
    # Hugging Face Methods
    # =========================================================================

    def push_to_hub(self, repo_id: str, private: bool = True):
        """Push model to Hugging Face Hub."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            super().push_to_hub(repo_id=repo_id, save_dir=tmp_dir, private=private)
            self.processor.tokenizer.push_to_hub(repo_id)

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, **kwargs):
        """
        Load model from Hugging Face Hub or local directory.
        
        To use a LoRA adapter:
            1. Load the base model first
            2. Then load the adapter using model.load_adapter()
        
        Example:
            model = Extractor.from_pretrained("base-model-name")
            model.load_adapter("path/to/adapter")
        """
        from huggingface_hub import hf_hub_download

        def download_or_local(repo, filename):
            if os.path.isdir(repo):
                return os.path.join(repo, filename)
            return hf_hub_download(repo, filename)

        config_path = download_or_local(repo_or_dir, "config.json")
        config = cls.config_class.from_pretrained(config_path)

        encoder_config_path = download_or_local(repo_or_dir, "encoder_config/config.json")
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)

        # Load schema encoder config for bi-encoder models
        schema_encoder_config = None
        if getattr(config, 'encoder_mode', 'uni') == 'bi':
            try:
                schema_enc_path = download_or_local(repo_or_dir, "schema_encoder_config/config.json")
                schema_encoder_config = AutoConfig.from_pretrained(schema_enc_path)
            except Exception:
                pass

        tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)

        # Load schema tokenizer for bi-encoder models
        schema_tokenizer = None
        if getattr(config, 'encoder_mode', 'uni') == 'bi':
            try:
                schema_tok_path = download_or_local(repo_or_dir, "schema_tokenizer/tokenizer_config.json")
                schema_tok_dir = os.path.dirname(schema_tok_path)
                schema_tokenizer = AutoTokenizer.from_pretrained(schema_tok_dir)
            except Exception:
                pass

        model = cls(config, encoder_config=encoder_config,
                    schema_encoder_config=schema_encoder_config,
                    tokenizer=tokenizer, schema_tokenizer=schema_tokenizer)

        # Load weights
        try:
            model_path = download_or_local(repo_or_dir, "model.safetensors")
            state_dict = load_file(model_path)
        except Exception:
            model_path = download_or_local(repo_or_dir, "pytorch_model.bin")
            state_dict = torch.load(model_path, map_location="cpu")

        # Handle embedding size mismatch
        def _fix_embedding_mismatch(state_dict, key, model_emb):
            try:
                saved_emb = state_dict[key]
                if saved_emb.shape[0] != model_emb.shape[0]:
                    extra = model_emb.shape[0] - saved_emb.shape[0]
                    state_dict[key] = torch.cat([
                        saved_emb,
                        torch.randn(extra, saved_emb.shape[1]) * 0.02
                    ], dim=0)
            except KeyError:
                pass

        def _get_embedding_key(encoder, prefix):
            """Find the embedding weight key for any encoder architecture."""
            emb = encoder.embeddings
            for name in ("word_embeddings", "tok_embeddings"):
                if hasattr(emb, name):
                    return f"{prefix}.embeddings.{name}.weight", getattr(emb, name).weight
            return None, None

        enc_key, enc_weight = _get_embedding_key(model.encoder, "encoder")
        if enc_key is not None:
            _fix_embedding_mismatch(state_dict, enc_key, enc_weight)

        if model.schema_encoder is not None and not model._shared_encoder:
            sch_key, sch_weight = _get_embedding_key(model.schema_encoder, "schema_encoder")
            if sch_key is not None:
                _fix_embedding_mismatch(state_dict, sch_key, sch_weight)

        load_result = model.load_state_dict(state_dict, strict=not model._shared_encoder)
        if load_result.missing_keys:
            print(f"[GLiNER2] WARNING: {len(load_result.missing_keys)} keys missing from checkpoint "
                  f"(using random init): {load_result.missing_keys[:5]}"
                  f"{'...' if len(load_result.missing_keys) > 5 else ''}")
        return model

    def load_adapter(self, adapter_path: str) -> 'Extractor':
        """
        Load a LoRA adapter onto this model.
        
        If an adapter is already loaded, it will be unloaded first.
        
        Args:
            adapter_path: Path to adapter directory
            
        Returns:
            self for method chaining
            
        Example:
            model.load_adapter("./legal_adapter")
            results = model.extract_entities(text, entities)
        """
        from gliner2.training.lora import load_lora_adapter, LoRAAdapterConfig
        
        # Load adapter config
        config = LoRAAdapterConfig.load(adapter_path)
        
        self._lora_layers = load_lora_adapter(self, adapter_path, auto_unload=True)
        self._adapter_config = config
        return self
    
    def unload_adapter(self) -> 'Extractor':
        """
        Unload current LoRA adapter, restoring base model.
        
        Returns:
            self for method chaining
        """
        from gliner2.training.lora import unload_lora_adapter
        
        if self._lora_layers:
            unload_lora_adapter(self)
            self._lora_layers = {}
            self._adapter_config = None
        return self
    
    def merge_lora(self) -> 'Extractor':
        """
        Merge LoRA weights into base model and remove adapter structure.
        
        After calling this, the model will have standard Linear layers with
        merged weights. LoRA adapters are permanently removed.
        
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If no adapter is loaded
            
        Example:
            model.load_adapter("./my_adapter")
            model.merge_lora()  # Now model has merged weights, no LoRA
            model.save_pretrained("./merged_model")
        """
        if not self._lora_layers:
            raise ValueError("No adapter loaded. Nothing to merge.")
        
        from gliner2.training.lora import merge_lora_weights
        merge_lora_weights(self)
        self._lora_layers = {}
        self._adapter_config = None
        return self
    
    def save_adapter(self, save_path: str) -> None:
        """
        Save only the LoRA adapter (not full model).
        
        Args:
            save_path: Directory to save adapter
            
        Raises:
            ValueError: If no adapter is loaded
        """
        if not self._lora_layers:
            raise ValueError("No adapter loaded. Use save_pretrained for full model.")
        
        from gliner2.training.lora import save_lora_adapter
        save_lora_adapter(self, save_path)
    
    @property
    def has_adapter(self) -> bool:
        """Check if an adapter is currently loaded."""
        return bool(self._lora_layers)
    
    @property
    def adapter_config(self):
        """Get config of loaded adapter, or None."""
        return self._adapter_config
    
    def save_pretrained(
        self, 
        save_directory: str,
        save_adapter_only: bool = False,
        merge_lora: bool = True,
        **kwargs
    ):
        """
        Save model to directory.
        
        Args:
            save_directory: Where to save
            save_adapter_only: If True and adapter loaded, save only adapter
            merge_lora: If True and LoRA active, merge LoRA weights into base
                       model and remove adapter structure before saving.
                       WARNING: This permanently removes LoRA from the model instance.
        """
        if save_adapter_only:
            if not self._lora_layers:
                raise ValueError("save_adapter_only=True but no adapter loaded")
            self.save_adapter(save_directory)
            return
        
        # Handle LoRA merging if requested
        if merge_lora and self._lora_layers:
            self.merge_lora()
        
        # Original save logic
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        if self.schema_encoder is not None and not self._shared_encoder:
            schema_enc_path = os.path.join(save_directory, "schema_encoder_config")
            os.makedirs(schema_enc_path, exist_ok=True)
            self.schema_encoder.config.save_pretrained(schema_enc_path)

        model_path = os.path.join(save_directory, "model.safetensors")
        sd = self.state_dict()
        if self._shared_encoder:
            sd = {k: v for k, v in sd.items() if not k.startswith("schema_encoder.")}
        save_file(sd, model_path)

        self.processor.tokenizer.save_pretrained(save_directory)

        if self.processor.schema_tokenizer is not None:
            schema_tok_path = os.path.join(save_directory, "schema_tokenizer")
            os.makedirs(schema_tok_path, exist_ok=True)
            self.processor.schema_tokenizer.save_pretrained(schema_tok_path)