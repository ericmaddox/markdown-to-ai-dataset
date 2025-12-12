#!/usr/bin/env python3
"""
markdown-to-ai-dataset
----------------------
Convert a folder of Markdown (.md) files to a JSONL/HuggingFace dataset for language model fine-tuning.

- Loads and chunks Markdown documents from a folder.
- Outputs:
    * HuggingFace split dataset (disk)
    * JSONL train/test splits for LLM training

Usage: (edit paths below, then run)
    python markdown-to-ai-dataset.py
"""

from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer  # (tokenizer step optional, see README)
import json

# Configuration
# -------------
data_folder = Path("data")           # Directory containing Markdown files
max_length = 512                     # Max characters per text chunk
stride = 256                         # Overlap (in chars) between adjacent chunks
train_test_split = 0.1               # Fraction for validation set
output_folder = Path("dataset_split") # Output dir for HuggingFace formatted dataset

# Step 1: Load Markdown files
all_texts = []
for md_file in sorted(data_folder.rglob("*.md")):
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
            all_texts.append(text)
print(f"Loaded {len(all_texts)} Markdown files.")
if not all_texts:
    raise FileNotFoundError(f"No .md files found in {data_folder}")

# Step 2: Chunk Documents
# Chunks are created with overlap to maximize training signal
examples = []
for text in all_texts:
    for i in range(0, len(text), stride):
        chunk = text[i:i + max_length]
        if len(chunk) < max_length // 4:
            continue  # Skip tiny chunks
        examples.append({"text": chunk})
print(f"Created {len(examples)} training chunks.")

# Step 3: Build and Split Dataset
# Uses HuggingFace Dataset for easy splitting and export
hf_dataset = Dataset.from_list(examples)
dataset = hf_dataset.train_test_split(test_size=train_test_split, seed=42)
print(f"Train set: {len(dataset['train'])}")
print(f"Validation set: {len(dataset['test'])}")

# Step 4: Write Dataset to Disk
# - HuggingFace dataset split
# - JSONL files for train/test
output_folder.mkdir(exist_ok=True)
dataset.save_to_disk(output_folder)
dataset["train"].to_json("dataset_train.json")
dataset["test"].to_json("dataset_test.json")
print(f"Saved HuggingFace dataset to '{output_folder}/'")
print("JSON files: dataset_train.json, dataset_test.json")

# Step 5: Show Example
print("\n--- Sample text chunk from training set ---")
print(dataset["train"][0]["text"][:200] + "…")
