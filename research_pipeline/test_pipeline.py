"""
Advanced Text-to-SQL Pipeline with Self-Correction and Multi-dimensional RAG
Implements:
1. Self-Correction Loop - Model fixes its own SQL errors
2. B5+B6 Combination - RAG + Enhanced schema prompts
3. Multi-dimensional RAG - Retrieves by question AND table similarity
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
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch/transformers not available.")

# ========== CONFIG ==========
REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "research_pipeline" / "cache" / "ecommerce_dw.duckdb"
TEST_DATA_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "test.csv"
TRAIN_DATA_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "train_merged.csv"
DB_CONTENT_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "db_content_samples.json"
RAG_INDEX_DIR = REPO_ROOT / "research_pipeline" / "rag_index"

ADAPTER_ID = "Ellbendls/Qwen-3-4b-Text_to_SQL"

MAX_SAMPLES = None
MAX_NEW_TOKENS = 256
MAX_TABLES = 8
MAX_CORRECTION_ATTEMPTS = 2  # Self-correction loop attempts
RAG_K = 5

# ========== SYSTEM PROMPTS ==========
SYSTEM_PROMPT_ENHANCED = """You are a SQL expert for TPC-DS Data Warehouse on DuckDB.

TPC-DS SCHEMA OVERVIEW:
- 3 Sales Channels: store_sales (ss_*), web_sales (ws_*), catalog_sales (cs_*)
- 3 Return Tables: store_returns (sr_*), web_returns (wr_*), catalog_returns (cr_*)
- Key Dimensions: customer (c_*), item (i_*), date_dim (d_*), store (s_*), warehouse (w_*)

COLUMN NAMING CONVENTION:
- *_sk = Surrogate Key (INTEGER, use for JOINs)
- *_id = Business ID (VARCHAR)
- ss_/ws_/cs_ = Store/Web/Catalog Sales prefixes
- d_year, d_moy, d_qoy = Year, Month of Year, Quarter of Year
- Gender info is in customer_demographics (cd_gender), NOT in customer table

COMMON JOINS:
- Sales -> Date: ON ss_sold_date_sk = d_date_sk
- Sales -> Item: ON ss_item_sk = i_item_sk
- Sales -> Customer: ON ss_customer_sk = c_customer_sk
- Customer -> Demographics: ON c_current_cdemo_sk = cd_demo_sk
- Customer -> Address: ON c_current_addr_sk = ca_address_sk

DUCKDB SPECIFIC:
- Use date_add() not dateadd()
- Use LIMIT not TOP
- String comparison is case-sensitive

Output ONLY valid SQL. No explanations."""

CORRECTION_PROMPT = """The SQL you generated has an error:
{error}

Please fix the SQL query. Output ONLY the corrected SQL, no explanations."""

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
    "gioi": "gender", "gioitinh": "gender", "tuoi": "age",
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
    # NEW: Add customer_demographics for gender/age queries
    if any(t in q_tokens for t in {"gender", "age", "education", "marital"}):
        ensure("customer_demographics"); ensure("customer")
    
    return selected[:max_tables]

# ========== ENHANCED SCHEMA TEXT ==========
TABLE_DESCRIPTIONS = {
    "store_sales": "Doanh số cửa hàng. Cột: ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_quantity, ss_net_paid",
    "customer": "Khách hàng. Cột: c_customer_sk, c_first_name, c_last_name, c_current_cdemo_sk (FK->demographics)",
    "customer_demographics": "Thông tin nhân khẩu. Cột: cd_demo_sk, cd_gender (M/F), cd_marital_status, cd_education_status",
    "item": "Sản phẩm. Cột: i_item_sk, i_item_id, i_item_desc, i_category, i_brand, i_current_price",
    "date_dim": "Thời gian. Cột: d_date_sk, d_date, d_year, d_moy (tháng), d_qoy (quý)",
    "store": "Cửa hàng. Cột: s_store_sk, s_store_name, s_state, s_city",
}

def build_enhanced_schema_text(tables: list[str], schema_map: dict) -> str:
    lines = []
    for table in tables:
        cols = schema_map.get(table, [])
        desc = TABLE_DESCRIPTIONS.get(table, "")
        if desc:
            lines.append(f"-- {desc}")
        lines.append(f"TABLE {table} (")
        for col, typ in cols:
            lines.append(f"  {col} {typ},")
        lines.append(")")
        lines.append("")
    return "\n".join(lines).strip()

# ========== SQL UTILS ==========
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

# ========== MULTI-DIMENSIONAL RAG ==========
def extract_tables_from_sql(sql: str) -> set[str]:
    """Extract table names from SQL query."""
    tables = set()
    # Match FROM/JOIN table names
    for match in re.finditer(r'\b(?:FROM|JOIN)\s+([a-z_]+)', sql, re.I):
        tables.add(match.group(1).lower())
    return tables

def load_rag_retriever():
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from research_pipeline.rag_retriever import TextToSQLRetriever
        if RAG_INDEX_DIR.exists():
            return TextToSQLRetriever.load(RAG_INDEX_DIR)
    except Exception as e:
        print(f"RAG Retriever load failed: {e}")
    return None

def retrieve_multi_dimensional(retriever, question: str, needed_tables: list[str], k: int = 5) -> list[dict]:
    """Multi-dimensional retrieval: by question similarity AND table overlap."""
    if retriever is None:
        return []
    
    # Get more candidates than needed
    candidates = retriever.retrieve(question, k=k*3)
    
    if not needed_tables:
        return candidates[:k]
    
    needed_set = set(needed_tables)
    
    # Re-rank by table overlap
    scored = []
    for c in candidates:
        sql_tables = extract_tables_from_sql(c.get("sql", ""))
        table_overlap = len(needed_set & sql_tables)
        # Combined score: 0.7 * semantic + 0.3 * table_overlap
        combined_score = 0.7 * c.get("score", 0) + 0.3 * (table_overlap / max(len(needed_set), 1))
        scored.append((combined_score, c))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:k]]

# ========== MODEL ==========
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if not HAS_TORCH:
        return None, None
    
    print(f"Loading adapter: {ADAPTER_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit = torch.cuda.is_available()
    
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, trust_remote_code=True)
    
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(ADAPTER_ID)
    base_model_id = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_id}")
    
    quant_config = BitsAndBytesConfig(load_in_4bit=True) if use_4bit else None
    model_kwargs = dict(
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"Model loaded on {device}")
    return tokenizer, model

def generate_sql_raw(prompt: str) -> str:
    if not HAS_TORCH or model is None:
        return "SELECT 1;"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return extract_sql(tokenizer.decode(gen_ids, skip_special_tokens=True))

def build_prompt(question: str, schema_text: str, examples: list = None, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_ENHANCED
        
    few_shot_text = ""
    if examples:
        few_shot_text = "EXAMPLES:\n"
        for ex in examples[:5]:  # Limit examples to avoid OOM
            few_shot_text += f"Q: {ex['question'][:100]}\nSQL: {ex['sql'][:200]}\n\n"
    
    user = f"SCHEMA:\n{schema_text}\n\n{few_shot_text}QUESTION:\n{question}\n\nSQL:"
    
    if tokenizer and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        )
    return f"{system_prompt}\n\n{user}"

def build_correction_prompt(question: str, schema_text: str, original_sql: str, error: str) -> str:
    """Build prompt for self-correction."""
    user = f"""SCHEMA:
{schema_text}

QUESTION:
{question}

YOUR PREVIOUS SQL:
{original_sql}

ERROR:
{error[:200]}

Please output ONLY the corrected SQL:"""
    
    if tokenizer and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT_ENHANCED}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT_ENHANCED}\n\n{user}"

# ========== SELF-CORRECTION LOOP ==========
def generate_sql_with_correction(question: str, schema_text: str, examples: list, con) -> tuple[str, int]:
    """Generate SQL with self-correction loop. Returns (sql, num_attempts)."""
    
    # First attempt
    prompt = build_prompt(question, schema_text, examples)
    gen_sql = generate_sql_raw(prompt)
    
    if not is_valid_sql(gen_sql):
        return gen_sql, 1
    
    # Try executing
    _, error = run_sql(con, gen_sql)
    
    if error is None:
        return gen_sql, 1  # Success on first try
    
    # Self-correction loop
    for attempt in range(MAX_CORRECTION_ATTEMPTS):
        correction_prompt = build_correction_prompt(question, schema_text, gen_sql, error)
        gen_sql = generate_sql_raw(correction_prompt)
        
        if not is_valid_sql(gen_sql):
            continue
        
        _, error = run_sql(con, gen_sql)
        if error is None:
            return gen_sql, attempt + 2  # Success after correction
    
    return gen_sql, MAX_CORRECTION_ATTEMPTS + 1

# ========== MAIN ==========
def main():
    print("="*50)
    print("Advanced Pipeline: Self-Correction + B5+B6 + Multi-RAG")
    print("="*50)
    
    # Setup
    con = setup_db()
    schema_map = build_schema_map(con)
    table_tokens = build_table_tokens(schema_map)
    
    # Load RAG
    retriever = load_rag_retriever()
    if retriever:
        print("RAG Retriever loaded")
    
    # Load test data
    if not TEST_DATA_PATH.exists():
        test_df = pd.read_csv(TRAIN_DATA_PATH)
    else:
        test_df = pd.read_csv(TEST_DATA_PATH)
    
    test_df = test_df.dropna(subset=["Transcription", "SQL Ground Truth"]).head(MAX_SAMPLES)
    print(f"Test samples: {len(test_df)}")
    
    # Load model
    load_model()
    
    # Run tests
    correct = 0
    valid_count = 0
    correction_stats = {1: 0, 2: 0, 3: 0}  # Track correction attempts
    
    for idx, row in test_df.iterrows():
        question = row["Transcription"]
        gt_sql = row["SQL Ground Truth"]
        
        # Dynamic table selection
        tables = select_tables(question, table_tokens, schema_map, MAX_TABLES)
        
        # Enhanced schema (B6)
        schema_text = build_enhanced_schema_text(tables, schema_map)
        
        # Multi-dimensional RAG (B5 enhanced)
        examples = retrieve_multi_dimensional(retriever, question, tables, k=RAG_K)
        
        # Generate with self-correction
        start = time.time()
        gen_sql, attempts = generate_sql_with_correction(question, schema_text, examples, con)
        gen_time = (time.time() - start) * 1000
        
        correction_stats[min(attempts, 3)] = correction_stats.get(min(attempts, 3), 0) + 1
        
        # Validate
        valid = is_valid_sql(gen_sql)
        if valid:
            valid_count += 1
        
        # Execute and compare
        gt_res, gt_err = run_sql(con, gt_sql)
        gen_res, gen_err = run_sql(con, gen_sql) if valid else (None, "INVALID")
        
        exec_match = False
        if not gt_err and not gen_err:
            keep = has_order_by(gt_sql) or has_order_by(gen_sql)
            gt_norm = normalize_rows(gt_res, keep)
            gen_norm = normalize_rows(gen_res, keep)
            exec_match = gt_norm == gen_norm
            if exec_match:
                correct += 1
        
        status = "OK" if exec_match else ("FIXED" if attempts > 1 and not gen_err else "ERR")
        print(f"  [{idx}] {status} Valid={valid}, Match={exec_match}, Attempts={attempts}, Time={gen_time:.0f}ms")
        if gen_err and not exec_match:
            print(f"      Error: {gen_err[:80]}")
    
    print(f"\n{'='*40}")
    print(f"Summary: Valid={valid_count}/{len(test_df)}, ExecMatch={correct}/{len(test_df)} ({100*correct/len(test_df):.1f}%)")
    print(f"Correction stats: 1st try={correction_stats.get(1,0)}, 2nd={correction_stats.get(2,0)}, 3rd={correction_stats.get(3,0)}")
    
    con.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
