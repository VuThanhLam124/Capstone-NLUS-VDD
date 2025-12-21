# Refactor Plan

## 1. Updated `generate_large_dataset.py`
- **Goal**: End-to-end data creation (Text -> Audio -> Tensor).
- **Steps**:
  1. Generate 100 queries (JSON).
  2. For each query:
     - Generate multiple audio variants (Voice + Augmentations).
     - Save as MP3 temp.
     - Load MP3, Resample to 16k, Save as Tensor (.pt).
     - Remove temp MP3 (optional, or keep).
  3. Create `metadata.csv` indexing all tensors.
- **Output Structure**:
  - `data/tensors/`: contains all `.pt` files.
  - `data/metadata.csv`: contains mapping.

## 2. Updated `kaggle_asr_pipeline.py`
- **Goal**: Pure Benchmark Runner.
- **Logic**:
  - Parse `--data_path` (default: `data/`).
  - Read `{data_path}/metadata.csv`.
  - Load Models.
  - Loop through dataframe:
    - Load tensor: `torch.load(os.path.join(data_path, row['tensor_filename']))`.
    - Transcribe.
    - Compute WER.
