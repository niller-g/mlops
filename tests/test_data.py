import os
import datasets
from src.mlops.data import download_data, preprocess_data


def test_download_data():
    raw_path = download_data(output_dir="data/raw")
    assert os.path.exists(raw_path), "Raw data directory not found."
    # Verify we can load the dataset
    dataset = datasets.load_from_disk(raw_path)
    assert len(dataset) > 0, "Dataset is empty"


def test_preprocess_data():
    # Ensure raw data exists before testing preprocessing
    raw_path = download_data(output_dir="data/raw")
    processed_path = preprocess_data(raw_path, output_dir="data/processed")
    assert os.path.exists(processed_path), "Processed data directory not found."
    # Verify we can load the processed dataset
    dataset = datasets.load_from_disk(processed_path)
    assert (
        "clean_text" in dataset.features
    ), "Processed dataset missing clean_text feature"
