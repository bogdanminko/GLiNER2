"""
Test: shared bi-encoder training and inference end-to-end.

Same encoder for text and schema (two forward passes, one model).
No separate schema_model_name → shared encoder by default.

Run:
    python tests/test_shared_bi_encoder.py
"""

import shutil
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, Classification, Relation
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from gliner2.model import Extractor, ExtractorConfig


OUTPUT_DIR = "./test_shared_bi_encoder_output"


def main():
    # 1. Create shared bi-encoder model
    print("=" * 60)
    print("Creating SHARED bi-encoder model...")
    print("=" * 60)

    model_config = ExtractorConfig(
        model_name="bert-base-uncased",
        max_width=8,
        counting_layer="count_lstm",
        token_pooling="first",
        encoder_mode="bi",
        # no schema_model_name → shared encoder
        # schema_projection_dim=256,
        cross_fuser_heads=8
    )
    model = Extractor(model_config)

    # Verify shared encoder
    print(f"\nShared encoder  : {model._shared_encoder}")
    print(f"Same object?    : {model.encoder is model.schema_encoder}")
    print(f"encoder id       : {id(model.encoder)}")
    print(f"schema_encoder id: {id(model.schema_encoder)}")
    print(f"Same memory addr : {id(model.encoder) == id(model.schema_encoder)}")
    print(f"Schema tokenizer: {model.processor.schema_tokenizer}")  # should be None
    print(f"text_proj       : {model.text_proj is not None}")
    print(f"schema_proj     : {model.schema_proj is not None}")

    # Build a sample batch and inspect
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset
    sample = InputExample(
        text="Apple CEO Tim Cook announced iPhone 15.",
        entities={"company": ["Apple"], "person": ["Tim Cook"], "product": ["iPhone 15"]},
    )
    dataset = ExtractorDataset([sample], validate=False)
    collator = ExtractorCollator(model.processor, is_training=False)
    batch = collator([dataset[0]])

    tok = model.processor.tokenizer
    print(f"\n--- Batch fields ---")
    print(f"input_ids shape        : {batch.input_ids.shape}  (text only)")
    print(f"schema_input_ids shape : {batch.schema_input_ids.shape}  (labels only)")

    text_decoded = tok.decode(batch.input_ids[0][batch.attention_mask[0].bool()], skip_special_tokens=False)
    schema_decoded = tok.decode(batch.schema_input_ids[0][batch.schema_attention_mask[0].bool()], skip_special_tokens=False)
    print(f"\nText encoder sees  : {text_decoded[:120]}...")
    print(f"Schema encoder sees: {schema_decoded[:120]}...")

    # 2. Training data
    examples = [
        InputExample(
            text="John Smith works at Google in California. The company is thriving and expanding rapidly.",
            entities={
                "person": ["John Smith"],
                "company": ["Google"],
                "location": ["California"],
            },
            classifications=[
                Classification(
                    task="sentiment",
                    labels=["positive", "negative", "neutral"],
                    true_label="positive",
                )
            ],
            relations=[
                Relation("works_at", head="John Smith", tail="Google"),
                Relation("located_in", head="Google", tail="California"),
            ],
        ),
        InputExample(
            text="Microsoft CEO Satya Nadella presented Azure updates in Seattle.",
            entities={
                "person": ["Satya Nadella"],
                "company": ["Microsoft"],
                "location": ["Seattle"],
                "product": ["Azure"],
            },
        ),
        InputExample(
            text="Elon Musk announced Tesla's new factory in Berlin, Germany.",
            entities={
                "person": ["Elon Musk"],
                "company": ["Tesla"],
                "location": ["Berlin", "Germany"],
            },
        ),
    ]

    # 3. Train
    print("\n" + "=" * 60)
    print("Training shared bi-encoder...")
    print("=" * 60)

    train_config = TrainingConfig(
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=2,
        encoder_lr=1e-5,
        task_lr=5e-4,
    )
    trainer = GLiNER2Trainer(model, train_config)
    trainer.train(train_data=examples)

    # 4. Reload and inference
    print("\n" + "=" * 60)
    print("Loading saved shared bi-encoder model...")
    print("=" * 60)

    model = GLiNER2.from_pretrained(f"{OUTPUT_DIR}/final")

    # Verify still shared after reload
    print(f"\nShared encoder after reload: {model._shared_encoder}")
    print(f"Same object after reload?  : {model.encoder is model.schema_encoder}")
    print(f"encoder id       : {id(model.encoder)}")
    print(f"schema_encoder id: {id(model.schema_encoder)}")
    print(f"Same memory addr : {id(model.encoder) == id(model.schema_encoder)}")

    text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
    entity_types = ["company", "person", "product", "location"]

    print(f"\nText: {text}")
    print(f"Entity types: {entity_types}")

    result = model.extract_entities(text, entity_types)
    print(f"Result: {result}")

    # 5. Cleanup
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Shared bi-encoder test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
