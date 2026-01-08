#!/usr/bin/env python3
"""
Benchmark SQLCoder (Defog) on TPC-DS Test Set
SQLCoder is a 15B model specialized for Text-to-SQL

Usage:
    # Download model first
    huggingface-cli download TheBloke/sqlcoder-GGUF sqlcoder.Q4_K_M.gguf --local-dir ./models
    
    # Run benchmark
    python benchmark_sqlcoder.py --easy --max-test-samples 15
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import duckdb

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("ERROR: llama-cpp-python not installed")
    print("Run: pip install llama-cpp-python")
    sys.exit(1)

# Import schema linking
try:
    from schema_linking import SchemaLinker
    HAS_SCHEMA_LINKING = True
except ImportError:
    HAS_SCHEMA_LINKING = False


# ========== CONFIG ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SQLCoder on TPC-DS")
    parser.add_argument("--model", type=str, default="./models/sqlcoder.Q4_K_M.gguf",
                        help="Path to GGUF model file")
    parser.add_argument("--test-data", type=str, default="research_pipeline/datasets/test.csv")
    parser.add_argument("--db", type=str, default="research_pipeline/cache/ecommerce_dw.duckdb")
    parser.add_argument("--easy", action="store_true", help="Use test_easy.csv")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--schema-linking", action="store_true")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Layers to offload to GPU (-1=all)")
    parser.add_argument("--ctx-size", type=int, default=8192, help="Context size")
    return parser.parse_args()


# ========== TPC-DS SCHEMA (SQLCoder format) ==========
TPCDS_SCHEMA = """
CREATE TABLE store_sales (
    ss_sold_date_sk BIGINT,
    ss_item_sk BIGINT,
    ss_customer_sk BIGINT,
    ss_cdemo_sk BIGINT,
    ss_hdemo_sk BIGINT,
    ss_store_sk BIGINT,
    ss_promo_sk BIGINT,
    ss_quantity INTEGER,
    ss_sales_price DECIMAL,
    ss_net_paid DECIMAL,
    ss_net_profit DECIMAL
);

CREATE TABLE store_returns (
    sr_returned_date_sk BIGINT,
    sr_item_sk BIGINT,
    sr_customer_sk BIGINT,
    sr_store_sk BIGINT,
    sr_reason_sk BIGINT,
    sr_return_quantity INTEGER,
    sr_return_amt DECIMAL,
    sr_net_loss DECIMAL
);

CREATE TABLE web_sales (
    ws_sold_date_sk BIGINT,
    ws_item_sk BIGINT,
    ws_bill_customer_sk BIGINT,
    ws_web_page_sk BIGINT,
    ws_web_site_sk BIGINT,
    ws_warehouse_sk BIGINT,
    ws_quantity INTEGER,
    ws_sales_price DECIMAL,
    ws_net_paid DECIMAL,
    ws_net_profit DECIMAL
);

CREATE TABLE web_returns (
    wr_returned_date_sk BIGINT,
    wr_item_sk BIGINT,
    wr_refunded_customer_sk BIGINT,
    wr_web_page_sk BIGINT,
    wr_reason_sk BIGINT,
    wr_return_quantity INTEGER,
    wr_return_amt DECIMAL,
    wr_net_loss DECIMAL
);

CREATE TABLE catalog_sales (
    cs_sold_date_sk BIGINT,
    cs_item_sk BIGINT,
    cs_bill_customer_sk BIGINT,
    cs_call_center_sk BIGINT,
    cs_warehouse_sk BIGINT,
    cs_quantity INTEGER,
    cs_sales_price DECIMAL,
    cs_net_paid DECIMAL,
    cs_net_profit DECIMAL
);

CREATE TABLE catalog_returns (
    cr_returned_date_sk BIGINT,
    cr_item_sk BIGINT,
    cr_refunded_customer_sk BIGINT,
    cr_call_center_sk BIGINT,
    cr_reason_sk BIGINT,
    cr_return_quantity INTEGER,
    cr_return_amount DECIMAL,
    cr_net_loss DECIMAL
);

CREATE TABLE inventory (
    inv_date_sk BIGINT,
    inv_item_sk BIGINT,
    inv_warehouse_sk BIGINT,
    inv_quantity_on_hand INTEGER
);

CREATE TABLE customer (
    c_customer_sk BIGINT PRIMARY KEY,
    c_customer_id VARCHAR,
    c_current_cdemo_sk BIGINT,
    c_current_hdemo_sk BIGINT,
    c_current_addr_sk BIGINT,
    c_first_name VARCHAR,
    c_last_name VARCHAR,
    c_email_address VARCHAR
);

CREATE TABLE customer_demographics (
    cd_demo_sk BIGINT PRIMARY KEY,
    cd_gender CHAR(1),
    cd_marital_status CHAR(1),
    cd_education_status VARCHAR,
    cd_credit_rating VARCHAR,
    cd_dep_count INTEGER
);

CREATE TABLE household_demographics (
    hd_demo_sk BIGINT PRIMARY KEY,
    hd_income_band_sk BIGINT,
    hd_buy_potential VARCHAR,
    hd_dep_count INTEGER,
    hd_vehicle_count INTEGER
);

CREATE TABLE customer_address (
    ca_address_sk BIGINT PRIMARY KEY,
    ca_city VARCHAR,
    ca_county VARCHAR,
    ca_state CHAR(2),
    ca_zip VARCHAR,
    ca_country VARCHAR
);

CREATE TABLE item (
    i_item_sk BIGINT PRIMARY KEY,
    i_item_id VARCHAR,
    i_item_desc VARCHAR,
    i_current_price DECIMAL,
    i_brand VARCHAR,
    i_class VARCHAR,
    i_category VARCHAR,
    i_manufact VARCHAR,
    i_color VARCHAR,
    i_size VARCHAR,
    i_product_name VARCHAR
);

CREATE TABLE date_dim (
    d_date_sk BIGINT PRIMARY KEY,
    d_date DATE,
    d_year INTEGER,
    d_moy INTEGER,
    d_dom INTEGER,
    d_qoy INTEGER,
    d_day_name VARCHAR,
    d_weekend CHAR(1)
);

CREATE TABLE store (
    s_store_sk BIGINT PRIMARY KEY,
    s_store_name VARCHAR,
    s_manager VARCHAR,
    s_city VARCHAR,
    s_state CHAR(2)
);

CREATE TABLE warehouse (
    w_warehouse_sk BIGINT PRIMARY KEY,
    w_warehouse_name VARCHAR,
    w_city VARCHAR,
    w_state CHAR(2)
);

CREATE TABLE web_page (
    wp_web_page_sk BIGINT PRIMARY KEY,
    wp_url VARCHAR,
    wp_type VARCHAR
);

CREATE TABLE call_center (
    cc_call_center_sk BIGINT PRIMARY KEY,
    cc_name VARCHAR,
    cc_manager VARCHAR
);

CREATE TABLE reason (
    r_reason_sk BIGINT PRIMARY KEY,
    r_reason_desc VARCHAR
);

CREATE TABLE promotion (
    p_promo_sk BIGINT PRIMARY KEY,
    p_promo_name VARCHAR,
    p_discount_active VARCHAR
);
"""

# SQLCoder prompt template
SQLCODER_PROMPT = """### Task
Generate a SQL query to answer the following question:
`{question}`

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers `{question}`:
```sql
"""


def build_prompt(question: str, schema: str = None) -> str:
    """Build SQLCoder prompt"""
    if schema is None:
        schema = TPCDS_SCHEMA
    return SQLCODER_PROMPT.format(question=question, schema=schema)


def postprocess_sql(sql: str) -> str:
    """Clean generated SQL"""
    # Extract SQL from response
    sql = sql.strip()
    
    # Remove markdown
    if "```" in sql:
        sql = sql.split("```")[0]
    
    # Remove trailing incomplete lines
    lines = sql.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('--'):
            clean_lines.append(line)
    sql = '\n'.join(clean_lines)
    
    # Fix common issues
    sql = re.sub(r'\bsr_sr_', 'sr_', sql)
    sql = re.sub(r'\bss_ss_', 'ss_', sql)
    sql = re.sub(r'\bws_ws_', 'ws_', sql)
    sql = re.sub(r'\bcs_cs_', 'cs_', sql)
    sql = re.sub(r'\br_r_', 'r_', sql)
    
    # Ensure ends with semicolon
    sql = sql.rstrip()
    if sql and not sql.endswith(';'):
        sql += ';'
    
    return sql


def main():
    args = parse_args()
    
    print("="*60)
    print("SQLCoder Benchmark on TPC-DS")
    print("="*60)
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"\nModel not found: {args.model}")
        print("\nDownload with:")
        print("  huggingface-cli download TheBloke/sqlcoder-GGUF sqlcoder.Q4_K_M.gguf --local-dir ./models")
        sys.exit(1)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    print("Model loaded!")
    
    # Load test data
    if args.easy:
        test_path = "research_pipeline/datasets/test_easy.csv"
    else:
        test_path = args.test_data
    
    test_df = pd.read_csv(test_path)
    print(f"\nTest set: {test_path} ({len(test_df)} samples)")
    
    if args.max_test_samples:
        test_df = test_df.head(args.max_test_samples)
        print(f"Using first {len(test_df)} samples")
    
    # Setup schema linker
    schema_linker = None
    if args.schema_linking and HAS_SCHEMA_LINKING:
        print("Loading schema linker...")
        schema_linker = SchemaLinker()
    
    # Setup DB
    conn = duckdb.connect(args.db, read_only=True)
    
    # Run benchmark
    results = []
    valid_count = 0
    exec_match_count = 0
    
    for idx, row in test_df.iterrows():
        # Handle different column names
        question = row.get('question') or row.get('Transcription')
        ground_truth = row.get('sql') or row.get('SQL Ground Truth')
        
        print(f"\n[{idx+1}/{len(test_df)}] {question[:60]}...")
        
        # Build prompt
        if schema_linker:
            schema = schema_linker.build_dynamic_schema(question, max_tables=5)
        else:
            schema = TPCDS_SCHEMA
        
        prompt = build_prompt(question, schema)
        
        # Generate
        start_time = time.time()
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["```", "\n\n\n"],
        )
        gen_time = (time.time() - start_time) * 1000
        
        generated_sql = postprocess_sql(output["choices"][0]["text"])
        
        # Validate
        is_valid = False
        exec_match = False
        error = None
        
        try:
            gen_result = conn.execute(generated_sql).fetchall()
            is_valid = True
            valid_count += 1
            
            try:
                gt_result = conn.execute(ground_truth).fetchall()
                if set(map(tuple, gen_result)) == set(map(tuple, gt_result)):
                    exec_match = True
                    exec_match_count += 1
                    print(f"  ✅ Match!")
                else:
                    print(f"  ⚠️ Different results")
            except Exception as e:
                print(f"  ⚠️ GT error: {e}")
                
        except Exception as e:
            error = str(e)
            print(f"  ❌ SQL Error: {error[:80]}")
        
        results.append({
            "id": idx,
            "question": question,
            "ground_truth": ground_truth,
            "generated_sql": generated_sql,
            "valid": is_valid,
            "exec_match": exec_match,
            "error": error,
            "gen_time_ms": gen_time,
        })
    
    conn.close()
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS - SQLCoder")
    print(f"{'='*60}")
    print(f"Valid SQL: {valid_count}/{len(test_df)} ({100*valid_count/len(test_df):.1f}%)")
    print(f"Exec Match: {exec_match_count}/{len(test_df)} ({100*exec_match_count/len(test_df):.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"research_pipeline/results/sqlcoder_benchmark_{timestamp}.json"
    Path("research_pipeline/results").mkdir(exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "SQLCoder-15B",
            "valid_sql": valid_count,
            "exec_match": exec_match_count,
            "total": len(test_df),
            "schema_linking": args.schema_linking,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
