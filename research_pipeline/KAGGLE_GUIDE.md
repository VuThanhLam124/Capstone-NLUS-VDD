# Kaggle Commands Guide - Text-to-SQL Pipeline

Hướng dẫn chạy Text-to-SQL Research Pipeline trên Kaggle với **BGE-M3 (Multilingual)** embedding.

## 🚀 Quick Setup (Copy all cells)

### Cell 1: Clone & Install Dependencies
```python
# Clone repo & install dependencies
!git clone https://github.com/VuThanhLam124/Capstone-NLUS-VDD.git 2>/dev/null || true
%cd Capstone-NLUS-VDD

# Install core dependencies
!pip install -q duckdb pandas scikit-learn sqlglot

# Install for RAG (BGE-M3 multilingual + BM25)
!pip install -q sentence-transformers rank-bm25

# Install for Text-to-SQL model
!pip install -q transformers peft accelerate bitsandbytes
```

### Cell 2: Build TPC-DS Database
```python
# Generate TPC-DS database (Scale Factor 1)
import duckdb
from pathlib import Path

db_path = Path('research_pipeline/cache/ecommerce_dw.duckdb')
db_path.parent.mkdir(parents=True, exist_ok=True)  # Create cache directory

con = duckdb.connect(str(db_path))
con.execute('INSTALL tpcds; LOAD tpcds;')
con.execute('CALL dsdgen(sf=1);')

# Verify
tables = con.execute('SHOW TABLES').fetchall()
print(f"✅ Database ready with {len(tables)} tables")
con.close()
```

### Cell 3: Build RAG Index (BGE-M3 + BM25 Hybrid)
```python
# Create rag_index directory first
!mkdir -p research_pipeline/rag_index

# Build RAG index with BGE-M3 multilingual embedding
!python research_pipeline/rag_retriever.py \
    --data research_pipeline/datasets/train_clean.csv \
    --output research_pipeline/rag_index
```

### Cell 4: Test RAG Retrieval
```python
# Test retrieval quality
import sys
sys.path.insert(0, 'research_pipeline')
from rag_retriever import TextToSQLRetriever

retriever = TextToSQLRetriever.load('research_pipeline/rag_index')

# Test Vietnamese queries
test_queries = [
    "Tính tổng doanh thu năm 2000",
    "Top 10 sản phẩm Electronics bán chạy nhất",
    "Khách hàng ở California mua nhiều nhất danh mục nào",
]

for q in test_queries:
    print(f"\n📝 Query: {q}")
    results = retriever.retrieve(q, k=3)
    for i, r in enumerate(results):
        print(f"   {i+1}. [score={r['score']:.3f}] {r['question'][:50]}...")
```

### Cell 5: Run Basic Pipeline Test (No Finetune)
```python
# Test pipeline without model inference (RAG + Schema Selection only)
import pandas as pd
from pathlib import Path

# Load test data
test_df = pd.read_csv('research_pipeline/datasets/test.csv')
test_df = test_df.dropna(subset=['Transcription', 'SQL Ground Truth']).head(20)

# Test schema selection
sys.path.insert(0, '.')
from research_pipeline.test_pipeline import SchemaSelector, build_schema_map, setup_db

con = setup_db()
schema_map = build_schema_map(con)
selector = SchemaSelector(schema_map)

print("Testing Schema Selection:")
for _, row in test_df.head(5).iterrows():
    q = row['Transcription']
    tables = selector.select_tables(q, max_tables=6)
    print(f"\n📝 {q[:50]}...")
    print(f"   Tables: {tables}")

con.close()
```

---

## 📦 File Structure
```
research_pipeline/
├── datasets/                    # Data files (committed)
│   ├── train_clean.csv          # Training data
│   ├── test.csv                 # Test data  
│   └── db_content_samples.json  # Schema enrichment
├── cache/                       # Generated at runtime
│   └── ecommerce_dw.duckdb      # TPC-DS database
├── rag_index/                   # RAG index (generated)
│   └── retriever_bge_m3.pkl     # BGE-M3 + BM25 index
├── rag_retriever.py             # RAG with BGE-M3 + BM25
├── test_pipeline.py             # Main pipeline
└── KAGGLE_GUIDE.md              # This file
```

---

## 🔧 Key Improvements

### 1. BGE-M3 Multilingual Embedding
- **Model**: `BAAI/bge-m3` (supports 100+ languages including Vietnamese)
- **Better than**: `BAAI/bge-large-en-v1.5` (English only)

### 2. Hybrid Retrieval
- **Semantic**: BGE-M3 cosine similarity (70%)
- **Lexical**: BM25 keyword matching (30%)
- **Combined**: `score = 0.7 * semantic + 0.3 * bm25`

### 3. Vietnamese Optimized
- No instruction prefix needed for BGE-M3
- Better handling of Vietnamese tokenization

---

## 📊 Expected Output

### RAG Retrieval Test:
```
Query: Tính tổng doanh thu năm 2000
  1. [score=0.823, sem=0.791, bm25=0.899] Tổng doanh thu store sales năm 2000...
  2. [score=0.756, sem=0.712, bm25=0.860] Doanh thu từ kênh web năm 2000...
  3. [score=0.721, sem=0.689, bm25=0.795] Tính revenue theo năm...
```

---

## ⚠️ Notes

1. **Memory**: BGE-M3 cần ~2GB GPU RAM
2. **Time**: Build RAG index ~5 phút trên Kaggle GPU
3. **No Finetune**: Pipeline hiện tại test RAG + Schema Selection, chưa có model inference

---

## 🔗 Links
- [BGE-M3 HuggingFace](https://huggingface.co/BAAI/bge-m3)
- [Sentence Transformers](https://www.sbert.net/)
- [TPC-DS DuckDB](https://duckdb.org/docs/extensions/tpcds)
