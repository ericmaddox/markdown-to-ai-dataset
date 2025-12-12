# markdown-to-ai-dataset

A Python utility to convert Markdown files into JSON datasets for AI language model fine-tuning, specifically formatted for use with the HuggingFace Datasets library.

## Features
- Converts a directory of Markdown (`.md`) files into a dataset for LLM training.
- Outputs HuggingFace-compatible JSON files for easy integration with common language model finetuning scripts.
- Randomly splits data into `dataset_train.json` (90%) and `dataset_test.json` (10%).
- Easy to customize for any markdown corpus.

## Quickstart

```bash
# Clone this repo or copy the folder
cd markdown-to-ai-dataset

# Install dependencies
pip install -r requirements.txt

# Place your markdown files in a folder, and set that folder path in the script
python markdown-to-ai-dataset.py
```

## Usage

- Edit `dataset_folder` in `markdown-to-ai-dataset.py` to point to your collection of `.md` files.
- The script will generate:
  - `dataset_train.json` — Main training split for language modeling
  - `dataset_test.json` — Test/validation split
- Each output JSON line looks like:
  ```json
  {"text": "<one document's markdown content here>"}
  ```

## Requirements

- Python 3.8 or newer
- `datasets`, `transformers`, `tqdm` (see `requirements.txt`)

## Example integration
After generating the datasets, you can use them with HuggingFace's Trainer utility or any fine-tuning pipeline expecting a JSONL corpus.

## License

MIT
