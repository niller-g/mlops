import os
import re
import logging
from pathlib import Path

import datasets

logging.basicConfig(level=logging.INFO)


def download_data(output_dir: str = "data/raw"):
    """
    Download the medical questions dataset and save locally.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Downloading dataset to {output_dir}...")

    # Using medical questions dataset from HF
    dataset = datasets.load_dataset("medical_questions_pairs", split="train")

    # Save it as a dataset
    raw_path = Path(output_dir) / "medical_questions_raw"
    dataset.save_to_disk(str(raw_path))
    logging.info(f"Dataset saved to {raw_path}")

    return str(raw_path)


def preprocess_data(input_path: str, output_dir: str = "data/processed"):
    """Clean up text, remove PII if needed, etc."""
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Reading raw data from {input_path}...")

    # Load the dataset
    dataset = datasets.load_from_disk(input_path)

    def clean_text(ex):
        # Combine question pairs into a single text
        text = f"Question 1: {ex['question_1']} Answer: {ex['question_2']}"
        # Simple cleaning:
        text = text.lower()
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return {"clean_text": text.strip()}

    # Create new dataset with only the clean text
    processed_dataset = dataset.map(clean_text)

    # Save processed dataset
    processed_path = Path(output_dir) / "medical_questions_processed"
    processed_dataset.save_to_disk(str(processed_path))
    logging.info(f"Processed dataset saved to {processed_path}")

    return str(processed_path)


if __name__ == "__main__":
    # Quick test
    raw = download_data()
    processed = preprocess_data(raw)
