#!/bin/bash
# Full Pipeline for Text-to-SQL with Schema Linking on vast.ai
# Run: bash research_pipeline/run_pipeline.sh

set -e  # Exit on error

echo "========================================="
echo "Text-to-SQL Pipeline with Schema Linking"
echo "========================================="

# Step 1: Install dependencies
echo -e "\n[1/5] Installing dependencies..."
pip install -q sentence-transformers rank-bm25

# Step 2: Test schema linking
echo -e "\n[2/5] Testing schema linking module..."
cd research_pipeline
python schema_linking.py
cd ..

# Step 3: Generate training data với schema linking
echo -e "\n[3/5] Generating TPC-DS training data with schema linking..."
python research_pipeline/generate_tpcds_training_data.py \
    --input research_pipeline/datasets/train_clean.csv \
    --output research_pipeline/datasets/train_tpcds_linked.jsonl

# Step 4: Benchmark WITHOUT schema linking (baseline)
echo -e "\n[4/5] Baseline benchmark (full schema, no linking)..."
python research_pipeline/finetune_and_benchmark.py \
    --skip-train \
    --adapter Ellbendls/Qwen-3-4b-Text_to_SQL \
    --easy \
    --few-shot 3 \
    --max-test-samples 10

# Step 5: Benchmark WITH schema linking
echo -e "\n[5/5] Schema linking benchmark (dynamic schema)..."
python research_pipeline/finetune_and_benchmark.py \
    --skip-train \
    --adapter Ellbendls/Qwen-3-4b-Text_to_SQL \
    --easy \
    --schema-linking \
    --few-shot 3 \
    --max-test-samples 10

echo -e "\n========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "Check results in: ./finetuned_model/benchmark_results.json"
