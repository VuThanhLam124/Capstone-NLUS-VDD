"""
Test Pipeline (No Finetuning Required)
Runs B0, B1, B4, B5 conditions on a small sample to verify correctness.
"""
import os
import sys
import json
import time
import re
import unicodedata
from pathlib import Path
from decimal import Decimal
from datetime import date, datetime
import math

import duckdb
import pandas as pd

# Try imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch/transformers not available. Using mock generation.")

# ========== CONFIG ==========
REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "research_pipeline" / "cache" / "ecommerce_dw.duckdb"
TEST_DATA_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "test.csv"
TRAIN_DATA_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "train_merged.csv"
DB_CONTENT_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "db_content_samples.json"
RAG_INDEX_DIR = REPO_ROOT / "research_pipeline" / "rag_index"

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # Production model
# Use pre-trained adapter from HuggingFace (no local finetuning needed)
ADAPTER_ID = "Ellbendls/Qwen-3-4b-Text_to_SQL"

MAX_SAMPLES = None  # Full dataset for Kaggle
MAX_NEW_TOKENS = 256
MAX_TABLES = 8

# Conditions to test (excluding B2, B3 which need finetuned model)
TEST_CONDITIONS = ["B0", "B1", "B4", "B5"]

# ========== SETUP DB ==========
def setup_db():
    if not DB_PATH.exists():
        print("Setting up TPC-DS database...")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(DB_PATH))
        con.execute("INSTALL tpcds; LOAD tpcds;")
        con.execute("CALL dsdgen(sf=1);")
        con.close()
    return duckdb.connect(str(DB_PATH), read_only=True)

# ========== SCHEMA UTILS ==========
def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")

def tokenize(text: str) -> list[str]:
    text = strip_accents(text.lower())
    tokens = []
    for tok in re.findall(r"[a-z0-9_]+", text):
        tokens.extend(tok.split("_"))
    return [t for t in tokens if len(t) > 1]

SYNONYMS = {
    "khach": "customer", "khachhang": "customer", "sanpham": "item",
    "hang": "item", "danhmuc": "category", "bang": "state", "tinh": "state",
    "cuahang": "store", "doanhthu": "revenue", "soluong": "quantity",
    "gia": "price", "thang": "month", "nam": "year", "quy": "quarter",
}

def expand_tokens(tokens: list[str]) -> set[str]:
    expanded = set(tokens)
    for tok in tokens:
        if tok in SYNONYMS:
            expanded.add(SYNONYMS[tok])
    return expanded

def build_schema_map(con):
    schema_map = {}
    for (table_name,) in con.execute("SHOW TABLES").fetchall():
        cols = [(r[0], r[1]) for r in con.execute(f"DESCRIBE {table_name}").fetchall()]
        schema_map[table_name] = cols
    return schema_map

def build_table_tokens(schema_map):
    table_tokens = {}
    for table, cols in schema_map.items():
        tokens = set(tokenize(table))
        for col, _ in cols:
            tokens.update(tokenize(col))
        table_tokens[table] = tokens
    return table_tokens

def select_tables(question: str, table_tokens: dict, schema_map: dict, max_tables: int = 8) -> list[str]:
    q_tokens = expand_tokens(tokenize(question))
    scored = [(len(q_tokens & tokens), table) for table, tokens in table_tokens.items()]
    scored.sort(reverse=True)
    selected = [t for score, t in scored if score > 0][:max_tables]
    
    def ensure(table):
        if table in schema_map and table not in selected:
            selected.append(table)
    
    if any(t in q_tokens for t in {"year", "month", "quarter", "date"}):
        ensure("date_dim")
    if "customer" in q_tokens:
        ensure("customer"); ensure("customer_address")
    if "store" in q_tokens:
        ensure("store_sales"); ensure("store")
    if any(t in q_tokens for t in {"sales", "revenue"}):
        ensure("store_sales")
    
    return selected[:max_tables]

def build_schema_text(tables: list[str], schema_map: dict, db_content: dict = None, with_content: bool = False) -> str:
    lines = []
    for table in tables:
        cols = schema_map.get(table, [])
        lines.append(f"TABLE {table} (")
        for col, typ in cols:
            extra = ""
            if with_content and db_content:
                samples = db_content.get(table, {}).get(col)
                if samples:
                    s_str = ", ".join(f"'{str(s)[:30]}'" for s in samples[:3])
                    extra = f" -- e.g. [{s_str}]"
            lines.append(f"  {col} {typ}{extra}")
        lines.append(")")
        lines.append("")
    return "\n".join(lines).strip()

# ========== SQL UTILS ==========
SYSTEM_PROMPT = "You translate user questions into SQL for DuckDB (TPC-DS). Return only SQL, no markdown."

def extract_sql(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.I | re.S)
    if m:
        text = m.group(1).strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    return text

_FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\b", re.I)

def is_valid_sql(sql: str) -> bool:
    s = re.sub(r"--.*$", "", sql, flags=re.M).strip()
    if not s or _FORBIDDEN.search(s):
        return False
    first = re.split(r"\s+", s, maxsplit=1)[0].upper()
    return first in {"SELECT", "WITH"}

def normalize_value(v):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, Decimal):
        return float(round(v, 6))
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v

def normalize_rows(rows, keep_order: bool):
    if rows is None:
        return None
    norm = [tuple(normalize_value(x) for x in row) for row in rows]
    if keep_order:
        return norm
    # Convert None to empty string for sorting (to avoid TypeError)
    def sort_key(row):
        return tuple("" if x is None else x for x in row)
    return sorted(norm, key=sort_key)

def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql, re.I))

def run_sql(con, sql: str):
    try:
        return con.execute(sql).fetchall(), None
    except Exception as e:
        return None, str(e)

# ========== RAG UTILS ==========
def load_rag_retriever():
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from research_pipeline.rag_retriever import TextToSQLRetriever
        if RAG_INDEX_DIR.exists():
            return TextToSQLRetriever.load(RAG_INDEX_DIR)
    except Exception as e:
        print(f"RAG Retriever load failed: {e}")
    return None

# ========== MODEL ==========
def load_model():
    if not HAS_TORCH:
        return None, None
    
    print(f"Loading model: {BASE_MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit = torch.cuda.is_available()
    
    quant_config = BitsAndBytesConfig(load_in_4bit=True) if use_4bit else None
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    model_kwargs = dict(
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"Model loaded on {device}")
    return tokenizer, model

def generate_sql(prompt: str, tokenizer, model) -> str:
    if not HAS_TORCH or model is None:
        # Mock generation for testing without GPU
        return "SELECT 1;"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return extract_sql(tokenizer.decode(gen_ids, skip_special_tokens=True))

def build_prompt(question: str, schema_text: str, tokenizer, examples: list = None) -> str:
    few_shot_text = ""
    if examples:
        few_shot_text = "EXAMPLES:\n"
        for ex in examples:
            few_shot_text += f"Q: {ex['question']}\nSQL: {ex['sql']}\n\n"
    
    user = f"SCHEMA:\n{schema_text}\n\n{few_shot_text}QUESTION:\n{question}\n\nSQL:"
    
    if tokenizer and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\n{user}"

# ========== MAIN ==========
def main():
    print("="*50)
    print("Pipeline Test (No Finetuning)")
    print("="*50)
    
    # Setup
    con = setup_db()
    schema_map = build_schema_map(con)
    table_tokens = build_table_tokens(schema_map)
    
    # Load DB Content (B4)
    db_content = {}
    if DB_CONTENT_PATH.exists():
        with open(DB_CONTENT_PATH, "r", encoding="utf-8") as f:
            db_content = json.load(f)
        print(f"Loaded content samples for {len(db_content)} tables")
    
    # Load RAG (B5)
    retriever = load_rag_retriever()
    if retriever:
        print("RAG Retriever loaded")
    
    # Load test data
    if not TEST_DATA_PATH.exists():
        print(f"Test data not found: {TEST_DATA_PATH}")
        print("Using train data instead...")
        test_df = pd.read_csv(TRAIN_DATA_PATH)
    else:
        test_df = pd.read_csv(TEST_DATA_PATH)
    
    test_df = test_df.dropna(subset=["Transcription", "SQL Ground Truth"]).head(MAX_SAMPLES)
    print(f"Test samples: {len(test_df)}")
    
    # Load model
    tokenizer, model = load_model()
    
    # Run tests
    for cond in TEST_CONDITIONS:
        print(f"\n{'='*40}\nCondition: {cond}\n{'='*40}")
        
        use_dynamic = cond in ["B1", "B4", "B5"]
        use_content = cond == "B4"
        use_rag = cond == "B5"
        
        correct = 0
        valid_count = 0
        
        for idx, row in test_df.iterrows():
            question = row["Transcription"]
            gt_sql = row["SQL Ground Truth"]
            
            # Build schema
            if use_dynamic:
                tables = select_tables(question, table_tokens, schema_map, MAX_TABLES)
            else:
                tables = list(schema_map.keys())
            schema_text = build_schema_text(tables, schema_map, db_content, with_content=use_content)
            
            # Get examples (RAG)
            examples = None
            if use_rag and retriever:
                examples = retriever.retrieve(question, k=3)
            
            # Generate
            prompt = build_prompt(question, schema_text, tokenizer, examples)
            start = time.time()
            gen_sql = generate_sql(prompt, tokenizer, model)
            gen_time = (time.time() - start) * 1000
            
            # Validate
            valid = is_valid_sql(gen_sql)
            if valid:
                valid_count += 1
            
            # Execute
            gt_res, gt_err = run_sql(con, gt_sql)
            gen_res, gen_err = run_sql(con, gen_sql) if valid else (None, "INVALID")
            
            # Compare
            exec_match = False
            if not gt_err and not gen_err:
                keep = has_order_by(gt_sql) or has_order_by(gen_sql)
                gt_norm = normalize_rows(gt_res, keep)
                gen_norm = normalize_rows(gen_res, keep)
                exec_match = gt_norm == gen_norm
                if exec_match:
                    correct += 1
            
            print(f"  [{idx}] Valid={valid}, ExecMatch={exec_match}, Time={gen_time:.0f}ms")
            if gen_err:
                print(f"      Error: {gen_err[:100]}")
        
        print(f"\n  Summary: Valid={valid_count}/{len(test_df)}, ExecMatch={correct}/{len(test_df)}")
    
    con.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
