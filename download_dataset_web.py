from datasets import load_dataset
import os

hf_dataset_name = "jesusmolrdv/MTF25-VLM-Challenge-Dataset-Web"
metadata_output_file = "mtf2025_web_metadata.parquet" # Output file for img2dataset

print(f"Loading dataset metadata: {hf_dataset_name}")
# Ensure you load the correct split, default is often 'train'
dataset = load_dataset(hf_dataset_name, split="train")

print(f"Saving metadata to {metadata_output_file}...")
# Save in Parquet format, which img2dataset can read efficiently
dataset.to_parquet(metadata_output_file)

print("Metadata file saved successfully.")
