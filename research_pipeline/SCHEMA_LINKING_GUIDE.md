# Schema Linking for Text-to-SQL (TPC-DS)

## 🎯 Overview

Hệ thống Schema Linking kết hợp 3 kỹ thuật state-of-the-art:

1. **Bidirectional Linking**: Question→Schema + Schema→Question
2. **Vector-based Retrieval**: Embedding similarity với BGE-M3
3. **Context Engineering**: Dynamic schema selection

## 📋 Setup

```bash
# Install dependencies
pip install sentence-transformers rank-bm25

# Test schema linking
python research_pipeline/schema_linking.py
```

## 🚀 Usage

### 1. Generate Training Data với Schema Linking

```bash
# With schema linking (recommended)
python research_pipeline/generate_tpcds_training_data.py \
    --input research_pipeline/datasets/train_clean.csv \
    --output research_pipeline/datasets/train_tpcds_linked.jsonl

# Without schema linking (full schema)
python research_pipeline/generate_tpcds_training_data.py --no-linking
```

### 2. Benchmark với Schema Linking

```bash
# Enable schema linking
python research_pipeline/finetune_and_benchmark.py \
    --skip-train \
    --adapter Ellbendls/Qwen-3-4b-Text_to_SQL \
    --easy \
    --schema-linking \
    --few-shot 3 \
    --max-test-samples 10

# Compare: Full schema vs Schema linking
python research_pipeline/finetune_and_benchmark.py --skip-train --easy  # Full schema
python research_pipeline/finetune_and_benchmark.py --skip-train --easy --schema-linking  # Linked
```

### 3. Finetune trên TPC-DS Data

```bash
# Step 1: Generate data
python research_pipeline/generate_tpcds_training_data.py

# Step 2: Train
python research_pipeline/finetune_and_benchmark.py \
    --train-data research_pipeline/datasets/train_tpcds_linked.jsonl \
    --adapter Ellbendls/Qwen-3-4b-Text_to_SQL \
    --output ./finetuned_tpcds \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-5

# Step 3: Evaluate
python research_pipeline/finetune_and_benchmark.py \
    --skip-train \
    --adapter ./finetuned_tpcds \
    --schema-linking \
    --easy
```

## 🔬 Kỹ Thuật

### Bidirectional Linking

```python
from schema_linking import SchemaLinker

linker = SchemaLinker()
result = linker.link_schema("Năm 2002 doanh thu bao nhiêu?")

print(result)
# {
#   'tables': ['store_sales', 'date_dim'],
#   'columns': ['ss_net_paid', 'd_year'],
#   'joins': ['JOIN date_dim ON ss_sold_date_sk = d_date_sk']
# }
```

### Dynamic Schema Generation

```python
# Instead of full 24 tables, only 3-5 relevant ones
schema = linker.build_dynamic_schema(
    "Sản phẩm bán chạy nhất",
    max_tables=3
)
# Returns:
# TABLE store_sales (ss_item_sk, ss_quantity, ss_net_paid)
# TABLE item (i_item_sk, i_product_name, i_category)
# JOIN HINTS:
#   JOIN item ON ss_item_sk = i_item_sk
```

## 📊 Expected Results

| Method | Context Size | Accuracy (Easy Set) |
|--------|--------------|---------------------|
| Full Schema (24 tables) | ~8000 tokens | ~10-20% |
| Schema Linking (3-5 tables) | ~1200 tokens | **30-40%** (expected) |
| + Few-shot (3 examples) | ~1500 tokens | **40-50%** (expected) |

## 🎓 Research References

- **RESDSQL** (Schema linking with representation learning)
- **DAIL-SQL** (Example selection via embedding)
- **DIN-SQL** (Decomposition and self-correction)

## 🔧 Troubleshooting

**Q: Schema linking không hoạt động?**
```bash
# Check dependencies
pip install sentence-transformers
python -c "from schema_linking import SchemaLinker; SchemaLinker()"
```

**Q: Accuracy vẫn thấp?**
- Finetune lại với `train_tpcds_linked.jsonl` (schema linking data)
- Tăng few-shot examples lên 5-7
- Thử model lớn hơn (7B/14B)

**Q: Training data có bao nhiêu samples?**
```bash
wc -l research_pipeline/datasets/train_tpcds_linked.jsonl
```

## 📝 Next Steps

1. **Generate training data**: `python generate_tpcds_training_data.py`
2. **Finetune model**: With `--train-data train_tpcds_linked.jsonl`
3. **Benchmark**: With `--schema-linking` flag
4. **Compare**: Full schema vs Linked schema vs Few-shot

## 📁 Generated Files

```
research_pipeline/datasets/
├── train_tpcds_linked.jsonl        # Training data with schema linking
├── train_tpcds_full_schema.jsonl  # Training data with full schema (fallback)
└── test_easy.csv                   # Test set (easier samples)
```
