# Plan: Enhance ASR Pipeline (1k Samples + Disfluencies)

## 1. Data Generation Update
- [x] Modify `generate_large_dataset.py` to produce **1000 samples**.
- [x] Inject Natural Speech patterns (Fillers: "ờ", "hmmm"; Prefixes: "Cho mình hỏi...").
- [x] Output file: `test_queries_vi_1000.json`.

## 2. Pipeline Script Update (`kaggle_asr_pipeline.py`)
- [ ] Rename script from `kaggle_pipeline.py` (Done).
- [ ] Update input path to `test_queries_vi_1000.json`.
- [ ] Ensure `generate_large_dataset.py` is called correctly if JSON missing.

## 3. Execution on Kaggle
User runs:
```python
!python research_pipeline/generate_large_dataset.py
!python research_pipeline/kaggle_asr_pipeline.py
```

## 4. Verification
- Use `head` command to check JSON file content for fillers.
- Run pipeline on a subset (local test) or full on Kaggle.
