# Pipeline Commands for vast.ai

## Môi trường đã setup + git clone ✅

### Thứ tự chạy:

```bash
# 1. Install dependencies (chỉ chạy 1 lần)
pip install sentence-transformers rank-bm25

# 2. Test schema linking
cd research_pipeline && python schema_linking.py && cd ..

# 3. Generate training data với schema linking
python research_pipeline/generate_tpcds_training_data.py \
    --input research_pipeline/datasets/train_clean.csv \
    --output research_pipeline/datasets/train_tpcds_linked.jsonl

# 4. Benchmark BASELINE (không có schema linking)
python research_pipeline/finetune_and_benchmark.py \
    --skip-train \
    --adapter Ellbendls/Qwen-3-4b-Text_to_SQL \
    --easy \
    --few-shot 3 \
    --max-test-samples 10

# 5. Benchmark WITH SCHEMA LINKING
python research_pipeline/finetune_and_benchmark.py \
    --skip-train \
    --adapter Ellbendls/Qwen-3-4b-Text_to_SQL \
    --easy \
    --schema-linking \
    --few-shot 3 \
    --max-test-samples 10

# 6. (Optional) Finetune trên TPC-DS data
python research_pipeline/finetune_and_benchmark.py \
    --train-data research_pipeline/datasets/train_tpcds_linked.jsonl \
    --adapter Ellbendls/Qwen-3-4b-Text_to_SQL \
    --output ./finetuned_tpcds \
    --epochs 3 \
    --batch-size 2

# 7. (Optional) Test finetuned model
python research_pipeline/finetune_and_benchmark.py \
    --skip-train \
    --adapter ./finetuned_tpcds \
    --easy \
    --schema-linking \
    --few-shot 3
```

## Hoặc chạy tất cả bằng 1 script:

```bash
bash research_pipeline/run_pipeline.sh
```
