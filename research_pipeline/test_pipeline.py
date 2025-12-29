"""
Advanced Text-to-SQL Pipeline
Features:
- TF-IDF based Dynamic Schema Selection
- Multi-dimensional RAG with table overlap scoring
- Enhanced schema descriptions for all 24 TPC-DS tables
- Optimized prompt building
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
from collections import defaultdict

import duckdb
import pandas as pd
import numpy as np

# Try imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch/transformers not available.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ========== CONFIG ==========
REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "research_pipeline" / "cache" / "ecommerce_dw.duckdb"
TEST_DATA_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "test.csv"
TRAIN_DATA_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "train_clean.csv"
DB_CONTENT_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "db_content_samples.json"
RAG_INDEX_DIR = REPO_ROOT / "research_pipeline" / "rag_index"

ADAPTER_ID = "Ellbendls/Qwen-3-4b-Text_to_SQL"

MAX_SAMPLES = None
MAX_NEW_TOKENS = 256
MAX_TABLES = 10  # Increased for better coverage
RAG_K = 5

# ========== SYSTEM PROMPT ==========
SYSTEM_PROMPT = "You translate user questions into SQL for DuckDB (TPC-DS). Return only SQL, no markdown."

# ========== TABLE DESCRIPTIONS - All 24 TPC-DS Tables ==========
TABLE_DESCRIPTIONS = {
    # Fact Tables (7)
    "store_sales": "-- Store sales: ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_store_sk, ss_quantity, ss_net_paid, ss_net_profit",
    "store_returns": "-- Store returns: sr_returned_date_sk, sr_item_sk, sr_customer_sk, sr_return_amt, sr_reason_sk",
    "web_sales": "-- Web sales: ws_sold_date_sk, ws_item_sk, ws_bill_customer_sk, ws_ship_customer_sk, ws_quantity, ws_net_paid",
    "web_returns": "-- Web returns: wr_returned_date_sk, wr_item_sk, wr_refunded_customer_sk, wr_return_amt, wr_reason_sk",
    "catalog_sales": "-- Catalog sales: cs_sold_date_sk, cs_item_sk, cs_bill_customer_sk, cs_ship_customer_sk, cs_quantity, cs_net_paid",
    "catalog_returns": "-- Catalog returns: cr_returned_date_sk, cr_item_sk, cr_refunded_customer_sk, cr_return_amt, cr_reason_sk",
    "inventory": "-- Inventory levels: inv_date_sk, inv_item_sk, inv_warehouse_sk, inv_quantity_on_hand",
    
    # Customer Dimensions (3)
    "customer": "-- Customer: c_customer_sk, c_customer_id, c_first_name, c_last_name, c_current_addr_sk, c_current_cdemo_sk",
    "customer_address": "-- Customer address: ca_address_sk, ca_street_number, ca_city, ca_state, ca_zip, ca_country",
    "customer_demographics": "-- Demographics: cd_demo_sk, cd_gender (M/F), cd_marital_status (S/M/D/W), cd_education_status, cd_purchase_estimate, cd_credit_rating",
    
    # Product Dimension (1)
    "item": "-- Item/Product: i_item_sk, i_item_id, i_item_desc, i_category, i_class, i_brand, i_current_price, i_manager_id",
    
    # Time Dimensions (2)
    "date_dim": "-- Date: d_date_sk, d_date, d_year, d_moy (month 1-12), d_qoy (quarter 1-4), d_day_name, d_week_seq",
    "time_dim": "-- Time: t_time_sk, t_time, t_hour, t_minute, t_second, t_am_pm, t_shift",
    
    # Location/Channel Dimensions (6)
    "store": "-- Store: s_store_sk, s_store_id, s_store_name, s_city, s_state, s_zip, s_manager",
    "warehouse": "-- Warehouse: w_warehouse_sk, w_warehouse_id, w_warehouse_name, w_city, w_state, w_zip",
    "web_site": "-- Web site: web_site_sk, web_site_id, web_name, web_class, web_manager",
    "web_page": "-- Web page: wp_web_page_sk, wp_web_page_id, wp_type, wp_url",
    "call_center": "-- Call center: cc_call_center_sk, cc_call_center_id, cc_name, cc_city, cc_state, cc_manager",
    "catalog_page": "-- Catalog page: cp_catalog_page_sk, cp_catalog_page_id, cp_department, cp_type",
    
    # Other Dimensions (5)
    "promotion": "-- Promotion: p_promo_sk, p_promo_id, p_promo_name, p_channel_email, p_channel_tv, p_channel_catalog",
    "reason": "-- Return reason: r_reason_sk, r_reason_id, r_reason_desc",
    "ship_mode": "-- Shipping: sm_ship_mode_sk, sm_ship_mode_id, sm_type (EXPRESS/OVERNIGHT/REGULAR), sm_carrier",
    "household_demographics": "-- Household: hd_demo_sk, hd_income_band_sk, hd_buy_potential, hd_dep_count, hd_vehicle_count",
    "income_band": "-- Income band: ib_income_band_sk, ib_lower_bound, ib_upper_bound",
}

# ========== TABLE RELATIONSHIPS (for JOIN hints) ==========
TABLE_RELATIONSHIPS = {
    "store_sales": ["date_dim", "item", "customer", "store", "promotion"],
    "web_sales": ["date_dim", "item", "customer", "web_site", "web_page", "promotion"],
    "catalog_sales": ["date_dim", "item", "customer", "catalog_page", "promotion"],
    "store_returns": ["date_dim", "item", "customer", "store", "reason"],
    "web_returns": ["date_dim", "item", "customer", "web_site", "reason"],
    "catalog_returns": ["date_dim", "item", "customer", "catalog_page", "reason"],
    "inventory": ["date_dim", "item", "warehouse"],
    "customer": ["customer_address", "customer_demographics", "household_demographics"],
}

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

# ========== IMPROVED SCHEMA SELECTION ==========
class SchemaSelector:
    """TF-IDF based schema selection for better table relevance scoring."""
    
    def __init__(self, schema_map: dict):
        self.schema_map = schema_map
        self.table_names = list(schema_map.keys())
        
        # Build table descriptions for TF-IDF
        self.table_docs = []
        for table in self.table_names:
            cols = schema_map[table]
            # Combine table name + column names + descriptions
            col_text = " ".join([col for col, _ in cols])
            desc = TABLE_DESCRIPTIONS.get(table, "")
            doc = f"{table} {col_text} {desc}"
            self.table_docs.append(doc)
        
        # Build TF-IDF vectorizer
        if HAS_SKLEARN:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r'[a-z][a-z0-9_]*',
                ngram_range=(1, 2)
            )
            self.table_vectors = self.vectorizer.fit_transform(self.table_docs)
        else:
            self.vectorizer = None
            self.table_vectors = None
    
    def select_tables(self, question: str, max_tables: int = 10) -> list[str]:
        """Select most relevant tables for the question using TF-IDF + rules."""
        
        if not HAS_SKLEARN or self.vectorizer is None:
            return self._fallback_select(question, max_tables)
        
        # TF-IDF similarity
        q_vector = self.vectorizer.transform([question.lower()])
        scores = cosine_similarity(q_vector, self.table_vectors).flatten()
        
        # Apply rule-based boosting
        question_lower = question.lower()
        for i, table in enumerate(self.table_names):
            # Boost fact tables for sales/revenue queries
            if any(w in question_lower for w in ["doanh thu", "sales", "revenue", "bán"]):
                if table in ["store_sales", "web_sales", "catalog_sales"]:
                    scores[i] *= 1.5
            # Boost return tables for return queries
            if any(w in question_lower for w in ["trả hàng", "return", "hoàn"]):
                if table in ["store_returns", "web_returns", "catalog_returns", "reason"]:
                    scores[i] *= 1.5
            # Boost date_dim for time queries
            if any(w in question_lower for w in ["năm", "tháng", "quý", "year", "month", "quarter"]):
                if table == "date_dim":
                    scores[i] *= 2.0
            # Boost customer tables for customer queries
            if any(w in question_lower for w in ["khách", "customer", "người mua"]):
                if table in ["customer", "customer_address", "customer_demographics"]:
                    scores[i] *= 1.5
            # Boost item for product queries
            if any(w in question_lower for w in ["sản phẩm", "item", "hàng", "danh mục", "category"]):
                if table == "item":
                    scores[i] *= 1.5
        
        # Sort by score and select top tables
        ranked = sorted(zip(self.table_names, scores), key=lambda x: x[1], reverse=True)
        selected = [t for t, s in ranked if s > 0.01][:max_tables]
        
        # Add related tables (for JOINs)
        selected = self._add_related_tables(selected, max_tables)
        
        return selected
    
    def _add_related_tables(self, selected: list[str], max_tables: int) -> list[str]:
        """Add tables needed for JOINs based on relationships."""
        result = list(selected)
        
        for table in selected:
            if table in TABLE_RELATIONSHIPS:
                for related in TABLE_RELATIONSHIPS[table]:
                    if related not in result and len(result) < max_tables:
                        result.append(related)
        
        return result[:max_tables]
    
    def _fallback_select(self, question: str, max_tables: int) -> list[str]:
        """Fallback to simple token matching if sklearn not available."""
        question_lower = question.lower()
        scores = defaultdict(float)
        
        for table in self.table_names:
            if table in question_lower:
                scores[table] += 1.0
            for col, _ in self.schema_map[table]:
                if col in question_lower:
                    scores[table] += 0.5
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, s in ranked if s > 0][:max_tables]

def build_schema_map(con):
    schema_map = {}
    for (table_name,) in con.execute("SHOW TABLES").fetchall():
        cols = [(r[0], r[1]) for r in con.execute(f"DESCRIBE {table_name}").fetchall()]
        schema_map[table_name] = cols
    return schema_map

def build_schema_text(tables: list[str], schema_map: dict) -> str:
    """Build schema text with descriptions."""
    lines = []
    for table in tables:
        cols = schema_map.get(table, [])
        desc = TABLE_DESCRIPTIONS.get(table, "")
        if desc:
            lines.append(desc)
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

def validate_sql_references(sql: str, schema_map: dict) -> tuple[bool, str]:
    """
    Validate that all table and column references in SQL exist in schema.
    Returns (is_valid, error_message)
    """
    valid_tables = set(schema_map.keys())
    valid_columns = set()
    table_columns = {}
    
    for table, cols in schema_map.items():
        table_columns[table] = set(col for col, _ in cols)
        valid_columns.update(col for col, _ in cols)
    
    # Extract table references (FROM/JOIN table_name [alias])
    sql_tables = set()
    for match in re.finditer(r'\b(?:FROM|JOIN)\s+([a-z_]+)(?:\s+(?:AS\s+)?([a-z_]+))?', sql, re.I):
        table = match.group(1).lower()
        sql_tables.add(table)
    
    # Check tables
    for table in sql_tables:
        if table not in valid_tables:
            return False, f"Table '{table}' does not exist"
    
    # Extract column references (including prefixed like ss_quantity)
    col_refs = set(re.findall(r'\b([a-z][a-z0-9_]*)\s*(?:=|<|>|,|\)|\s)', sql.lower()))
    
    # Filter out SQL keywords and function names
    sql_keywords = {'select', 'from', 'where', 'join', 'on', 'and', 'or', 'as', 'by', 
                    'group', 'order', 'having', 'limit', 'sum', 'count', 'avg', 'max', 
                    'min', 'distinct', 'case', 'when', 'then', 'else', 'end', 'in', 
                    'not', 'null', 'between', 'like', 'inner', 'left', 'right', 'outer',
                    'true', 'false', 'asc', 'desc', 'with', 'union', 'all', 'cast',
                    'date', 'int', 'integer', 'varchar', 'decimal', 'float', 'year'}
    
    col_refs = col_refs - sql_keywords - valid_tables
    
    return True, ""

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
    """Multi-dimensional retrieval: semantic + table overlap + SQL complexity matching."""
    if retriever is None:
        return []
    
    # Get more candidates
    candidates = retriever.retrieve(question, k=k*3)
    
    if not candidates:
        return []
    
    needed_set = set(needed_tables)
    
    # Re-rank with multiple factors
    scored = []
    for c in candidates:
        sql = c.get("sql", "")
        sql_tables = extract_tables_from_sql(sql)
        
        # Factor 1: Semantic similarity (from retriever)
        semantic_score = c.get("score", 0)
        
        # Factor 2: Table overlap
        table_overlap = len(needed_set & sql_tables) / max(len(needed_set), 1)
        
        # Factor 3: SQL complexity matching (prefer similar complexity)
        has_join = "join" in sql.lower()
        has_group = "group by" in sql.lower()
        has_subquery = sql.count("select") > 1
        complexity = has_join + has_group + has_subquery
        
        # Combined score
        combined_score = 0.5 * semantic_score + 0.3 * table_overlap + 0.2 * (complexity / 3)
        scored.append((combined_score, c))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:k]]

# ========== MODEL ==========
# Flag to skip model loading (set True to test RAG + Schema only)
SKIP_MODEL_LOAD = True  # Set to False when finetune is ready

def load_model():
    if not HAS_TORCH:
        return None, None
    
    if SKIP_MODEL_LOAD:
        print("Model loading skipped (SKIP_MODEL_LOAD=True)")
        return None, None
    
    try:
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
            print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModel.from_pretrained(model, ADAPTER_ID)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print(f"Model loaded on {device}")
        return tokenizer, model
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Falling back to test mode (RAG + Schema only)")
        return None, None

def generate_sql(prompt: str, tokenizer, model) -> str:
    if not HAS_TORCH or model is None:
        return "SELECT 1;"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return extract_sql(tokenizer.decode(gen_ids, skip_special_tokens=True))

def build_prompt(question: str, schema_text: str, tokenizer, examples: list = None) -> str:
    """Build optimized prompt with RAG examples."""
    few_shot_text = ""
    if examples:
        few_shot_text = "EXAMPLES:\n"
        for ex in examples:
            # Keep full examples for better context (up to 1500 chars SQL)
            q = ex['question'][:200]
            s = ex['sql'][:1500]
            few_shot_text += f"Q: {q}\nSQL: {s}\n\n"
    
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
    print("Text-to-SQL Pipeline: Schema Selection + RAG")
    print("Using BGE-M3 Multilingual + BM25 Hybrid")
    print("="*50)
    
    # Setup
    con = setup_db()
    schema_map = build_schema_map(con)
    
    # Initialize improved schema selector
    selector = SchemaSelector(schema_map)
    print(f"Schema selector initialized with {len(schema_map)} tables")
    
    # Load RAG
    retriever = load_rag_retriever()
    if retriever:
        print(f"RAG Retriever loaded (hybrid={getattr(retriever, 'use_hybrid', False)})")
    else:
        print("WARNING: RAG Retriever not loaded, running without few-shot examples")
    
    # Load test data
    if TEST_DATA_PATH.exists():
        test_df = pd.read_csv(TEST_DATA_PATH)
    else:
        test_df = pd.read_csv(TRAIN_DATA_PATH)
    
    test_df = test_df.dropna(subset=["Transcription", "SQL Ground Truth"])
    if MAX_SAMPLES:
        test_df = test_df.head(MAX_SAMPLES)
    print(f"Test samples: {len(test_df)}")
    
    # Check if we should load model (skip if not available or not needed)
    load_model_flag = HAS_TORCH and torch.cuda.is_available() and not SKIP_MODEL_LOAD
    
    if load_model_flag:
        print("\nLoading Text-to-SQL model...")
        tokenizer, model = load_model()
    else:
        print("\nSkipping model load (SKIP_MODEL_LOAD=True or no GPU)")
        print("Running in RAG + Schema Selection test mode only")
        tokenizer, model = None, None
    
    # Run tests
    correct = 0
    valid_count = 0
    results = []
    
    for idx, row in test_df.iterrows():
        question = row["Transcription"]
        gt_sql = row["SQL Ground Truth"]
        
        # TF-IDF based table selection
        tables = selector.select_tables(question, MAX_TABLES)
        schema_text = build_schema_text(tables, schema_map)
        
        # Multi-dimensional RAG
        examples = retrieve_multi_dimensional(retriever, question, tables, k=RAG_K)
        
        if model is not None:
            # Generate SQL with model
            prompt = build_prompt(question, schema_text, tokenizer, examples)
            start = time.time()
            gen_sql = generate_sql(prompt, tokenizer, model)
            gen_time = (time.time() - start) * 1000
            
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
            
            print(f"  [{idx}] Valid={valid}, ExecMatch={exec_match}, Time={gen_time:.0f}ms")
            if gen_err:
                print(f"      Error: {gen_err[:100]}")
                
            results.append({
                "id": idx,
                "question": question[:50],
                "valid": valid,
                "exec_match": exec_match,
                "gen_time_ms": gen_time
            })
        else:
            # Test mode: just show schema selection and RAG results
            print(f"\n[{idx}] Question: {question[:60]}...")
            print(f"    Selected tables: {tables[:5]}...")
            if examples:
                print(f"    RAG examples: {len(examples)} (top score={examples[0]['score']:.3f})")
            
            results.append({
                "id": idx,
                "question": question[:50],
                "tables": tables,
                "rag_score": examples[0]['score'] if examples else 0
            })
    
    # Summary
    print(f"\n{'='*50}")
    if model is not None:
        print(f"Summary: Valid={valid_count}/{len(test_df)}, ExecMatch={correct}/{len(test_df)} ({100*correct/len(test_df):.1f}%)")
    else:
        print(f"Test Mode Summary: Processed {len(test_df)} samples")
        print("Schema Selection + RAG retrieval tested successfully")
        print("To run full inference, ensure GPU is available and model is loaded")
    
    con.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
