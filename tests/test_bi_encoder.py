"""
Test: bi-encoder training and inference end-to-end.

Creates a bi-encoder Extractor (mmBERT-small + paraphrase-multilingual-MiniLM),
trains for a few epochs on toy data, saves, reloads, and runs inference.

Run:
    python tests/test_bi_encoder.py
"""

import shutil
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, Classification, Relation
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from gliner2.model import Extractor, ExtractorConfig


OUTPUT_DIR = "./test_bi_encoder_output"


def main():
    # 1. Create bi-encoder model
    print("=" * 60)
    print("Creating bi-encoder model...")
    print("=" * 60)

    model_config = ExtractorConfig(
        model_name="jhu-clsp/mmBERT-small",
        max_width=8,
        counting_layer="count_lstm",
        token_pooling="first",
        encoder_mode="bi",
        schema_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        schema_projection_dim=256,
    )
    model = Extractor(model_config)

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
    print("Training...")
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
    print("Loading saved model...")
    print("=" * 60)

    model = GLiNER2.from_pretrained(f"{OUTPUT_DIR}/final")

    text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
    entity_types = ["company", "person", "product", "location"]

    print(f"\nText: {text}")
    print(f"Entity types: {entity_types}")

    result = model.extract_entities(text, entity_types)
    print(f"Result: {result}")

    # 5. Cleanup
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Bi-encoder test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
