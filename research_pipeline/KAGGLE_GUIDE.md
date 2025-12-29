# Kaggle Commands Guide

Hướng dẫn chạy Text-to-SQL Research Pipeline trên Kaggle.

## Setup (Cell 1)
```python
# Clone repo & install dependencies
!git clone https://github.com/VuThanhLam124/Capstone-NLUS-VDD.git
%cd Capstone-NLUS-VDD
!pip install -q duckdb pandas scikit-learn sqlglot transformers peft accelerate bitsandbytes
```

## Step 1: Build TPC-DS Database (Cell 2)
```python
# Generate TPC-DS database (Scale Factor 1, ~1GB data)
!python -c "
import duckdb
from pathlib import Path
db_path = Path('research_pipeline/cache/ecommerce_dw.duckdb')
db_path.parent.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(str(db_path))
con.execute('INSTALL tpcds; LOAD tpcds;')
con.execute('CALL dsdgen(sf=1);')
con.close()
print('Database generated!')
"
```

## Step 2: Build RAG Index (Cell 3)
```python
# Build TF-IDF RAG index from training data
!python research_pipeline/rag_retriever.py
```

## Step 3: Run Benchmark (Cell 4)
Chạy 4 conditions (B0, B1, B4, B5) - không cần finetune:
```python
!python research_pipeline/test_pipeline.py 2>&1 | tee benchmark_results.log
```

## [Optional] Step 4: Run Full Benchmark Notebook
Mở notebook Jupyter và chạy từng cell:
```
notebooks/research_benchmark.ipynb
```
Notebook này hỗ trợ 6 conditions (B0-B5), bao gồm cả B2/B3 cần adapter fine-tuned.

---

## File Structure
```
research_pipeline/
├── datasets/              # [COMMIT] Data files
│   ├── train_merged.csv   # 3,720 training samples
│   ├── test.csv           # 90 test samples
│   ├── dev.csv            # 90 dev samples
│   └── db_content_samples.json  # Schema enrichment
├── cache/                 # [GITIGNORED] Generated at runtime
│   └── ecommerce_dw.duckdb
├── rag_index/             # RAG retriever index
├── test_pipeline.py       # Quick test script (B0, B1, B4, B5)
├── rag_retriever.py       # TF-IDF retriever
└── dump_db_content.py     # Schema content extractor
```
