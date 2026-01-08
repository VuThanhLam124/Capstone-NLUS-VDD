"""
Combined Finetune and Benchmark Script
Trains model and immediately benchmarks on test set.
"""
import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
import torch
import duckdb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# Import schema linking
try:
    from schema_linking import SchemaLinker
    HAS_SCHEMA_LINKING = True
except ImportError:
    HAS_SCHEMA_LINKING = False
    print("WARNING: schema_linking.py not found - using full schema")

# ========== CONFIG ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune and Benchmark Text-to-SQL")
    
    # Data paths
    parser.add_argument("--train-data", type=str, default="research_pipeline/datasets/train_merged_final.jsonl",
                        help="Training data (JSONL)")
    parser.add_argument("--test-data", type=str, default="research_pipeline/datasets/test.csv",
                        help="Test data (CSV)")
    parser.add_argument("--db", type=str, default="research_pipeline/cache/ecommerce_dw.duckdb",
                        help="Database path")
    
    # Model
    parser.add_argument("--adapter", type=str, default="Ellbendls/Qwen-3-4b-Text_to_SQL",
                        help="Base adapter to finetune from")
    parser.add_argument("--output", type=str, default="./finetuned_model",
                        help="Output directory")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    
    # Benchmark params
    parser.add_argument("--max-test-samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only benchmark")
    parser.add_argument("--easy", action="store_true", help="Use easy test set (test_easy.csv)")
    parser.add_argument("--few-shot", type=int, default=0, help="Number of few-shot examples (0-3)")
    parser.add_argument("--base-model", type=str, default=None, 
                        help="Use base model directly (e.g., Qwen/Qwen2.5-3B-Instruct) instead of adapter")
    parser.add_argument("--schema-linking", action="store_true", 
                        help="Use dynamic schema linking (instead of full schema)")
    
    return parser.parse_args()

# ========== VALID TABLES ==========
VALID_TABLES = {
    'store_sales', 'store_returns', 'web_sales', 'web_returns', 
    'catalog_sales', 'catalog_returns', 'inventory',
    'customer', 'customer_address', 'customer_demographics',
    'item', 'date_dim', 'time_dim', 'store', 'warehouse',
    'web_site', 'web_page', 'call_center', 'catalog_page',
    'promotion', 'reason', 'ship_mode', 'household_demographics', 'income_band'
}

# ========== DATA LOADING ==========
def validate_sql_sample(sql: str) -> bool:
    tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-z_]+)', sql, re.I))
    for table in tables:
        if table.lower() not in VALID_TABLES:
            return False
    if re.search(r'(\.[a-z]+){4,}', sql):
        return False
    return True

def load_train_data(data_path: str, tokenizer) -> Dataset:
    samples = []
    skipped = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("messages", [])
            
            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
            if not validate_sql_sample(assistant_msg):
                skipped += 1
                continue
            
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if len(tokenizer.encode(text)) > 1800:
                skipped += 1
                continue
            
            samples.append({"text": text})
    
    print(f"Loaded {len(samples)} valid samples, skipped {skipped}")
    return Dataset.from_list(samples)

# ========== TRAINING ==========
def train_model(args, tokenizer, model):
    print("\n" + "="*60)
    print("PHASE 1: FINETUNING")
    print("="*60)
    
    # Load data
    dataset = load_train_data(args.train_data, tokenizer)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check tensorboard
    try:
        import tensorboard
        report_to = "tensorboard"
    except ImportError:
        report_to = "none"
    
    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_ratio=0.1,
        logging_steps=10,
        logging_dir=str(output_dir / "logs"),
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=report_to,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to: {args.output}")
    
    return model

# ========== BENCHMARKING ==========
def setup_db(db_path: str):
    if not Path(db_path).exists():
        print("Setting up TPC-DS database...")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(db_path)
        con.execute("INSTALL tpcds; LOAD tpcds;")
        con.execute("CALL dsdgen(sf=1);")
        con.close()
    return duckdb.connect(db_path, read_only=True)

def run_sql(con, sql):
    try:
        return con.execute(sql).fetchall(), None
    except Exception as e:
        return None, str(e)

def extract_sql(text: str) -> str:
    # Remove Qwen3 thinking blocks
    if '</think>' in text:
        text = text.split('</think>')[-1]
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    text = re.sub(r'^```sql\s*', '', text.strip())
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'```$', '', text)
    
    # Find first SELECT or WITH statement
    match = re.search(r'\b(SELECT|WITH)\b', text, re.IGNORECASE)
    if match:
        text = text[match.start():]
    
    if ';' in text:
        text = text[:text.index(';')+1]
    
    return text.strip()

def postprocess_sql(sql: str) -> str:
    """Convert SQL Server syntax to DuckDB syntax (NOT fixing model errors)"""
    # Fix TOP N -> LIMIT N (SQL Server to DuckDB dialect)
    top_match = re.search(r'\bSELECT\s+TOP\s+(\d+)\b', sql, re.IGNORECASE)
    if top_match:
        n = top_match.group(1)
        sql = re.sub(r'\bSELECT\s+TOP\s+\d+\b', 'SELECT', sql, flags=re.IGNORECASE)
        if 'LIMIT' not in sql.upper():
            sql = sql.rstrip(';').strip() + f' LIMIT {n};'
    
    # SQL Server -> DuckDB function mappings
    sql = re.sub(r'\bgetdate\s*\(\s*\)', 'CURRENT_DATE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bISNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)
    
    return sql

def generate_sql(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    raw_sql = extract_sql(raw_output)
    return postprocess_sql(raw_sql)

# ========== TPC-DS SCHEMA (Compact) ==========
TPCDS_SCHEMA = """
### Tables and Columns (TPC-DS E-commerce Schema)

**store_sales** (ss): ss_sold_date_sk, ss_sold_time_sk, ss_item_sk, ss_customer_sk, ss_cdemo_sk, ss_hdemo_sk, ss_addr_sk, ss_store_sk, ss_promo_sk, ss_ticket_number, ss_quantity, ss_wholesale_cost, ss_list_price, ss_sales_price, ss_ext_discount_amt, ss_ext_sales_price, ss_ext_wholesale_cost, ss_ext_list_price, ss_ext_tax, ss_coupon_amt, ss_net_paid, ss_net_paid_inc_tax, ss_net_profit

**store_returns** (sr): sr_returned_date_sk, sr_return_time_sk, sr_item_sk, sr_customer_sk, sr_cdemo_sk, sr_hdemo_sk, sr_addr_sk, sr_store_sk, sr_reason_sk, sr_ticket_number, sr_return_quantity, sr_return_amt, sr_return_tax, sr_return_amt_inc_tax, sr_fee, sr_return_ship_cost, sr_refunded_cash, sr_reversed_charge, sr_store_credit, sr_net_loss

**web_sales** (ws): ws_sold_date_sk, ws_sold_time_sk, ws_ship_date_sk, ws_item_sk, ws_bill_customer_sk, ws_bill_cdemo_sk, ws_bill_hdemo_sk, ws_bill_addr_sk, ws_ship_customer_sk, ws_ship_cdemo_sk, ws_ship_hdemo_sk, ws_ship_addr_sk, ws_web_page_sk, ws_web_site_sk, ws_ship_mode_sk, ws_warehouse_sk, ws_promo_sk, ws_order_number, ws_quantity, ws_wholesale_cost, ws_list_price, ws_sales_price, ws_ext_discount_amt, ws_ext_sales_price, ws_ext_wholesale_cost, ws_ext_list_price, ws_ext_tax, ws_coupon_amt, ws_ext_ship_cost, ws_net_paid, ws_net_paid_inc_tax, ws_net_paid_inc_ship, ws_net_paid_inc_ship_tax, ws_net_profit

**web_returns** (wr): wr_returned_date_sk, wr_returned_time_sk, wr_item_sk, wr_refunded_customer_sk, wr_refunded_cdemo_sk, wr_refunded_hdemo_sk, wr_refunded_addr_sk, wr_returning_customer_sk, wr_returning_cdemo_sk, wr_returning_hdemo_sk, wr_returning_addr_sk, wr_web_page_sk, wr_reason_sk, wr_order_number, wr_return_quantity, wr_return_amt, wr_return_tax, wr_return_amt_inc_tax, wr_fee, wr_return_ship_cost, wr_refunded_cash, wr_reversed_charge, wr_account_credit, wr_net_loss

**catalog_sales** (cs): cs_sold_date_sk, cs_sold_time_sk, cs_ship_date_sk, cs_bill_customer_sk, cs_bill_cdemo_sk, cs_bill_hdemo_sk, cs_bill_addr_sk, cs_ship_customer_sk, cs_ship_cdemo_sk, cs_ship_hdemo_sk, cs_ship_addr_sk, cs_call_center_sk, cs_catalog_page_sk, cs_ship_mode_sk, cs_warehouse_sk, cs_item_sk, cs_promo_sk, cs_order_number, cs_quantity, cs_wholesale_cost, cs_list_price, cs_sales_price, cs_ext_discount_amt, cs_ext_sales_price, cs_ext_wholesale_cost, cs_ext_list_price, cs_ext_tax, cs_coupon_amt, cs_ext_ship_cost, cs_net_paid, cs_net_paid_inc_tax, cs_net_paid_inc_ship, cs_net_paid_inc_ship_tax, cs_net_profit

**catalog_returns** (cr): cr_returned_date_sk, cr_returned_time_sk, cr_item_sk, cr_refunded_customer_sk, cr_refunded_cdemo_sk, cr_refunded_hdemo_sk, cr_refunded_addr_sk, cr_returning_customer_sk, cr_returning_cdemo_sk, cr_returning_hdemo_sk, cr_returning_addr_sk, cr_call_center_sk, cr_catalog_page_sk, cr_ship_mode_sk, cr_warehouse_sk, cr_reason_sk, cr_order_number, cr_return_quantity, cr_return_amount, cr_return_tax, cr_return_amt_inc_tax, cr_fee, cr_return_ship_cost, cr_refunded_cash, cr_reversed_charge, cr_store_credit, cr_net_loss

**inventory** (inv): inv_date_sk, inv_item_sk, inv_warehouse_sk, inv_quantity_on_hand

**customer** (c): c_customer_sk, c_customer_id, c_current_cdemo_sk, c_current_hdemo_sk, c_current_addr_sk, c_first_shipto_date_sk, c_first_sales_date_sk, c_salutation, c_first_name, c_last_name, c_preferred_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, c_login, c_email_address, c_last_review_date_sk

**customer_address** (ca): ca_address_sk, ca_address_id, ca_street_number, ca_street_name, ca_street_type, ca_suite_number, ca_city, ca_county, ca_state, ca_zip, ca_country, ca_gmt_offset, ca_location_type

**customer_demographics** (cd): cd_demo_sk, cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate, cd_credit_rating, cd_dep_count, cd_dep_employed_count, cd_dep_college_count

**item** (i): i_item_sk, i_item_id, i_rec_start_date, i_rec_end_date, i_item_desc, i_current_price, i_wholesale_cost, i_brand_id, i_brand, i_class_id, i_class, i_category_id, i_category, i_manufact_id, i_manufact, i_size, i_formulation, i_color, i_units, i_container, i_manager_id, i_product_name

**date_dim** (d): d_date_sk, d_date_id, d_date, d_month_seq, d_week_seq, d_quarter_seq, d_year, d_dow, d_moy, d_dom, d_qoy, d_fy_year, d_fy_quarter_seq, d_fy_week_seq, d_day_name, d_quarter_name, d_holiday, d_weekend, d_following_holiday, d_first_dom, d_last_dom, d_same_day_ly, d_same_day_lq, d_current_day, d_current_week, d_current_month, d_current_quarter, d_current_year

**time_dim** (t): t_time_sk, t_time_id, t_time, t_hour, t_minute, t_second, t_am_pm, t_shift, t_sub_shift, t_meal_time

**store** (s): s_store_sk, s_store_id, s_rec_start_date, s_rec_end_date, s_closed_date_sk, s_store_name, s_number_employees, s_floor_space, s_hours, s_manager, s_market_id, s_geography_class, s_market_desc, s_market_manager, s_division_id, s_division_name, s_company_id, s_company_name, s_street_number, s_street_name, s_street_type, s_suite_number, s_city, s_county, s_state, s_zip, s_country, s_gmt_offset, s_tax_percentage

**warehouse** (w): w_warehouse_sk, w_warehouse_id, w_warehouse_name, w_warehouse_sq_ft, w_street_number, w_street_name, w_street_type, w_suite_number, w_city, w_county, w_state, w_zip, w_country, w_gmt_offset

**web_site** (web): web_site_sk, web_site_id, web_rec_start_date, web_rec_end_date, web_name, web_open_date_sk, web_close_date_sk, web_class, web_manager, web_mkt_id, web_mkt_class, web_mkt_desc, web_market_manager, web_company_id, web_company_name, web_street_number, web_street_name, web_street_type, web_suite_number, web_city, web_county, web_state, web_zip, web_country, web_gmt_offset, web_tax_percentage

**web_page** (wp): wp_web_page_sk, wp_web_page_id, wp_rec_start_date, wp_rec_end_date, wp_creation_date_sk, wp_access_date_sk, wp_autogen_flag, wp_customer_sk, wp_url, wp_type, wp_char_count, wp_link_count, wp_image_count, wp_max_ad_count

**call_center** (cc): cc_call_center_sk, cc_call_center_id, cc_rec_start_date, cc_rec_end_date, cc_closed_date_sk, cc_open_date_sk, cc_name, cc_class, cc_employees, cc_sq_ft, cc_hours, cc_manager, cc_mkt_id, cc_mkt_class, cc_mkt_desc, cc_market_manager, cc_division, cc_division_name, cc_company, cc_company_name, cc_street_number, cc_street_name, cc_street_type, cc_suite_number, cc_city, cc_county, cc_state, cc_zip, cc_country, cc_gmt_offset, cc_tax_percentage

**catalog_page** (cp): cp_catalog_page_sk, cp_catalog_page_id, cp_start_date_sk, cp_end_date_sk, cp_department, cp_catalog_number, cp_catalog_page_number, cp_description, cp_type

**promotion** (p): p_promo_sk, p_promo_id, p_start_date_sk, p_end_date_sk, p_item_sk, p_cost, p_response_target, p_promo_name, p_channel_dmail, p_channel_email, p_channel_catalog, p_channel_tv, p_channel_radio, p_channel_press, p_channel_event, p_channel_demo, p_channel_details, p_purpose, p_discount_active

**reason** (r): r_reason_sk, r_reason_id, r_reason_desc

**ship_mode** (sm): sm_ship_mode_sk, sm_ship_mode_id, sm_type, sm_code, sm_carrier, sm_contract

**household_demographics** (hd): hd_demo_sk, hd_income_band_sk, hd_buy_potential, hd_dep_count, hd_vehicle_count

**income_band** (ib): ib_income_band_sk, ib_lower_bound, ib_upper_bound

### Key Relationships
- Sales tables (*_sales) link to dimensions via *_sk foreign keys
- date_dim.d_date_sk links to *_sold_date_sk, *_ship_date_sk, *_returned_date_sk
- customer.c_customer_sk links to *_customer_sk
- item.i_item_sk links to *_item_sk
- Use d_year from date_dim for year filtering (NOT year() function on date keys)
"""

# Compact schema format matching training data
TPCDS_SCHEMA_COMPACT = """TABLE store_sales (ss_sold_date_sk BIGINT, ss_sold_time_sk BIGINT, ss_item_sk BIGINT, ss_customer_sk BIGINT, ss_cdemo_sk BIGINT, ss_hdemo_sk BIGINT, ss_addr_sk BIGINT, ss_store_sk BIGINT, ss_promo_sk BIGINT, ss_ticket_number BIGINT, ss_quantity BIGINT, ss_wholesale_cost DECIMAL, ss_list_price DECIMAL, ss_sales_price DECIMAL, ss_ext_discount_amt DECIMAL, ss_ext_sales_price DECIMAL, ss_ext_wholesale_cost DECIMAL, ss_ext_list_price DECIMAL, ss_ext_tax DECIMAL, ss_coupon_amt DECIMAL, ss_net_paid DECIMAL, ss_net_paid_inc_tax DECIMAL, ss_net_profit DECIMAL)
TABLE store_returns (sr_returned_date_sk BIGINT, sr_return_time_sk BIGINT, sr_item_sk BIGINT, sr_customer_sk BIGINT, sr_cdemo_sk BIGINT, sr_hdemo_sk BIGINT, sr_addr_sk BIGINT, sr_store_sk BIGINT, sr_reason_sk BIGINT, sr_ticket_number BIGINT, sr_return_quantity BIGINT, sr_return_amt DECIMAL, sr_return_tax DECIMAL, sr_return_amt_inc_tax DECIMAL, sr_fee DECIMAL, sr_return_ship_cost DECIMAL, sr_refunded_cash DECIMAL, sr_reversed_charge DECIMAL, sr_store_credit DECIMAL, sr_net_loss DECIMAL)
TABLE web_sales (ws_sold_date_sk BIGINT, ws_sold_time_sk BIGINT, ws_ship_date_sk BIGINT, ws_item_sk BIGINT, ws_bill_customer_sk BIGINT, ws_bill_cdemo_sk BIGINT, ws_bill_hdemo_sk BIGINT, ws_bill_addr_sk BIGINT, ws_ship_customer_sk BIGINT, ws_ship_cdemo_sk BIGINT, ws_ship_hdemo_sk BIGINT, ws_ship_addr_sk BIGINT, ws_web_page_sk BIGINT, ws_web_site_sk BIGINT, ws_ship_mode_sk BIGINT, ws_warehouse_sk BIGINT, ws_promo_sk BIGINT, ws_order_number BIGINT, ws_quantity BIGINT, ws_wholesale_cost DECIMAL, ws_list_price DECIMAL, ws_sales_price DECIMAL, ws_ext_discount_amt DECIMAL, ws_ext_sales_price DECIMAL, ws_ext_wholesale_cost DECIMAL, ws_ext_list_price DECIMAL, ws_ext_tax DECIMAL, ws_coupon_amt DECIMAL, ws_ext_ship_cost DECIMAL, ws_net_paid DECIMAL, ws_net_paid_inc_tax DECIMAL, ws_net_paid_inc_ship DECIMAL, ws_net_paid_inc_ship_tax DECIMAL, ws_net_profit DECIMAL)
TABLE web_returns (wr_returned_date_sk BIGINT, wr_returned_time_sk BIGINT, wr_item_sk BIGINT, wr_refunded_customer_sk BIGINT, wr_refunded_cdemo_sk BIGINT, wr_refunded_hdemo_sk BIGINT, wr_refunded_addr_sk BIGINT, wr_returning_customer_sk BIGINT, wr_returning_cdemo_sk BIGINT, wr_returning_hdemo_sk BIGINT, wr_returning_addr_sk BIGINT, wr_web_page_sk BIGINT, wr_reason_sk BIGINT, wr_order_number BIGINT, wr_return_quantity BIGINT, wr_return_amt DECIMAL, wr_return_tax DECIMAL, wr_return_amt_inc_tax DECIMAL, wr_fee DECIMAL, wr_return_ship_cost DECIMAL, wr_refunded_cash DECIMAL, wr_reversed_charge DECIMAL, wr_account_credit DECIMAL, wr_net_loss DECIMAL)
TABLE catalog_sales (cs_sold_date_sk BIGINT, cs_sold_time_sk BIGINT, cs_ship_date_sk BIGINT, cs_bill_customer_sk BIGINT, cs_bill_cdemo_sk BIGINT, cs_bill_hdemo_sk BIGINT, cs_bill_addr_sk BIGINT, cs_ship_customer_sk BIGINT, cs_ship_cdemo_sk BIGINT, cs_ship_hdemo_sk BIGINT, cs_ship_addr_sk BIGINT, cs_call_center_sk BIGINT, cs_catalog_page_sk BIGINT, cs_ship_mode_sk BIGINT, cs_warehouse_sk BIGINT, cs_item_sk BIGINT, cs_promo_sk BIGINT, cs_order_number BIGINT, cs_quantity BIGINT, cs_wholesale_cost DECIMAL, cs_list_price DECIMAL, cs_sales_price DECIMAL, cs_ext_discount_amt DECIMAL, cs_ext_sales_price DECIMAL, cs_ext_wholesale_cost DECIMAL, cs_ext_list_price DECIMAL, cs_ext_tax DECIMAL, cs_coupon_amt DECIMAL, cs_ext_ship_cost DECIMAL, cs_net_paid DECIMAL, cs_net_paid_inc_tax DECIMAL, cs_net_paid_inc_ship DECIMAL, cs_net_paid_inc_ship_tax DECIMAL, cs_net_profit DECIMAL)
TABLE catalog_returns (cr_returned_date_sk BIGINT, cr_returned_time_sk BIGINT, cr_item_sk BIGINT, cr_refunded_customer_sk BIGINT, cr_refunded_cdemo_sk BIGINT, cr_refunded_hdemo_sk BIGINT, cr_refunded_addr_sk BIGINT, cr_returning_customer_sk BIGINT, cr_returning_cdemo_sk BIGINT, cr_returning_hdemo_sk BIGINT, cr_returning_addr_sk BIGINT, cr_call_center_sk BIGINT, cr_catalog_page_sk BIGINT, cr_ship_mode_sk BIGINT, cr_warehouse_sk BIGINT, cr_reason_sk BIGINT, cr_order_number BIGINT, cr_return_quantity BIGINT, cr_return_amount DECIMAL, cr_return_tax DECIMAL, cr_return_amt_inc_tax DECIMAL, cr_fee DECIMAL, cr_return_ship_cost DECIMAL, cr_refunded_cash DECIMAL, cr_reversed_charge DECIMAL, cr_store_credit DECIMAL, cr_net_loss DECIMAL)
TABLE inventory (inv_date_sk BIGINT, inv_item_sk BIGINT, inv_warehouse_sk BIGINT, inv_quantity_on_hand INTEGER)
TABLE customer (c_customer_sk BIGINT, c_customer_id VARCHAR, c_current_cdemo_sk BIGINT, c_current_hdemo_sk BIGINT, c_current_addr_sk BIGINT, c_first_shipto_date_sk BIGINT, c_first_sales_date_sk BIGINT, c_salutation VARCHAR, c_first_name VARCHAR, c_last_name VARCHAR, c_preferred_cust_flag VARCHAR, c_birth_day BIGINT, c_birth_month BIGINT, c_birth_year BIGINT, c_birth_country VARCHAR, c_login VARCHAR, c_email_address VARCHAR, c_last_review_date_sk INTEGER)
TABLE customer_address (ca_address_sk BIGINT, ca_address_id VARCHAR, ca_street_number VARCHAR, ca_street_name VARCHAR, ca_street_type VARCHAR, ca_suite_number VARCHAR, ca_city VARCHAR, ca_county VARCHAR, ca_state VARCHAR, ca_zip VARCHAR, ca_country VARCHAR, ca_gmt_offset DECIMAL, ca_location_type VARCHAR)
TABLE customer_demographics (cd_demo_sk BIGINT, cd_gender VARCHAR, cd_marital_status VARCHAR, cd_education_status VARCHAR, cd_purchase_estimate BIGINT, cd_credit_rating VARCHAR, cd_dep_count BIGINT, cd_dep_employed_count BIGINT, cd_dep_college_count INTEGER)
TABLE item (i_item_sk BIGINT, i_item_id VARCHAR, i_rec_start_date DATE, i_rec_end_date DATE, i_item_desc VARCHAR, i_current_price DECIMAL, i_wholesale_cost DECIMAL, i_brand_id BIGINT, i_brand VARCHAR, i_class_id BIGINT, i_class VARCHAR, i_category_id BIGINT, i_category VARCHAR, i_manufact_id BIGINT, i_manufact VARCHAR, i_size VARCHAR, i_formulation VARCHAR, i_color VARCHAR, i_units VARCHAR, i_container VARCHAR, i_manager_id BIGINT, i_product_name VARCHAR)
TABLE date_dim (d_date_sk BIGINT, d_date_id VARCHAR, d_date DATE, d_month_seq BIGINT, d_week_seq BIGINT, d_quarter_seq BIGINT, d_year BIGINT, d_dow BIGINT, d_moy BIGINT, d_dom BIGINT, d_qoy BIGINT, d_fy_year BIGINT, d_fy_quarter_seq BIGINT, d_fy_week_seq BIGINT, d_day_name VARCHAR, d_quarter_name VARCHAR, d_holiday VARCHAR, d_weekend VARCHAR, d_following_holiday VARCHAR, d_first_dom BIGINT, d_last_dom BIGINT, d_same_day_ly BIGINT, d_same_day_lq BIGINT, d_current_day VARCHAR, d_current_week VARCHAR, d_current_month VARCHAR, d_current_quarter VARCHAR, d_current_year VARCHAR)
TABLE time_dim (t_time_sk BIGINT, t_time_id VARCHAR, t_time BIGINT, t_hour BIGINT, t_minute BIGINT, t_second BIGINT, t_am_pm VARCHAR, t_shift VARCHAR, t_sub_shift VARCHAR, t_meal_time VARCHAR)
TABLE store (s_store_sk BIGINT, s_store_id VARCHAR, s_rec_start_date DATE, s_rec_end_date DATE, s_closed_date_sk BIGINT, s_store_name VARCHAR, s_number_employees BIGINT, s_floor_space BIGINT, s_hours VARCHAR, s_manager VARCHAR, s_market_id BIGINT, s_geography_class VARCHAR, s_market_desc VARCHAR, s_market_manager VARCHAR, s_division_id BIGINT, s_division_name VARCHAR, s_company_id BIGINT, s_company_name VARCHAR, s_street_number VARCHAR, s_street_name VARCHAR, s_street_type VARCHAR, s_suite_number VARCHAR, s_city VARCHAR, s_county VARCHAR, s_state VARCHAR, s_zip VARCHAR, s_country VARCHAR, s_gmt_offset DECIMAL, s_tax_percentage DECIMAL)
TABLE warehouse (w_warehouse_sk BIGINT, w_warehouse_id VARCHAR, w_warehouse_name VARCHAR, w_warehouse_sq_ft BIGINT, w_street_number VARCHAR, w_street_name VARCHAR, w_street_type VARCHAR, w_suite_number VARCHAR, w_city VARCHAR, w_county VARCHAR, w_state VARCHAR, w_zip VARCHAR, w_country VARCHAR, w_gmt_offset DECIMAL)
TABLE web_site (web_site_sk BIGINT, web_site_id VARCHAR, web_rec_start_date DATE, web_rec_end_date DATE, web_name VARCHAR, web_open_date_sk BIGINT, web_close_date_sk BIGINT, web_class VARCHAR, web_manager VARCHAR, web_mkt_id BIGINT, web_mkt_class VARCHAR, web_mkt_desc VARCHAR, web_market_manager VARCHAR, web_company_id BIGINT, web_company_name VARCHAR, web_street_number VARCHAR, web_street_name VARCHAR, web_street_type VARCHAR, web_suite_number VARCHAR, web_city VARCHAR, web_county VARCHAR, web_state VARCHAR, web_zip VARCHAR, web_country VARCHAR, web_gmt_offset DECIMAL, web_tax_percentage DECIMAL)
TABLE web_page (wp_web_page_sk BIGINT, wp_web_page_id VARCHAR, wp_rec_start_date DATE, wp_rec_end_date DATE, wp_creation_date_sk BIGINT, wp_access_date_sk BIGINT, wp_autogen_flag VARCHAR, wp_customer_sk BIGINT, wp_url VARCHAR, wp_type VARCHAR, wp_char_count BIGINT, wp_link_count BIGINT, wp_image_count BIGINT, wp_max_ad_count INTEGER)
TABLE call_center (cc_call_center_sk BIGINT, cc_call_center_id VARCHAR, cc_rec_start_date DATE, cc_rec_end_date DATE, cc_closed_date_sk BIGINT, cc_open_date_sk BIGINT, cc_name VARCHAR, cc_class VARCHAR, cc_employees BIGINT, cc_sq_ft BIGINT, cc_hours VARCHAR, cc_manager VARCHAR, cc_mkt_id BIGINT, cc_mkt_class VARCHAR, cc_mkt_desc VARCHAR, cc_market_manager VARCHAR, cc_division BIGINT, cc_division_name VARCHAR, cc_company BIGINT, cc_company_name VARCHAR, cc_street_number VARCHAR, cc_street_name VARCHAR, cc_street_type VARCHAR, cc_suite_number VARCHAR, cc_city VARCHAR, cc_county VARCHAR, cc_state VARCHAR, cc_zip VARCHAR, cc_country VARCHAR, cc_gmt_offset DECIMAL, cc_tax_percentage DECIMAL)
TABLE catalog_page (cp_catalog_page_sk BIGINT, cp_catalog_page_id VARCHAR, cp_start_date_sk BIGINT, cp_end_date_sk BIGINT, cp_department VARCHAR, cp_catalog_number BIGINT, cp_catalog_page_number BIGINT, cp_description VARCHAR, cp_type VARCHAR)
TABLE promotion (p_promo_sk BIGINT, p_promo_id VARCHAR, p_start_date_sk BIGINT, p_end_date_sk BIGINT, p_item_sk BIGINT, p_cost DECIMAL, p_response_target BIGINT, p_promo_name VARCHAR, p_channel_dmail VARCHAR, p_channel_email VARCHAR, p_channel_catalog VARCHAR, p_channel_tv VARCHAR, p_channel_radio VARCHAR, p_channel_press VARCHAR, p_channel_event VARCHAR, p_channel_demo VARCHAR, p_channel_details VARCHAR, p_purpose VARCHAR, p_discount_active VARCHAR)
TABLE reason (r_reason_sk BIGINT, r_reason_id VARCHAR, r_reason_desc VARCHAR)
TABLE ship_mode (sm_ship_mode_sk BIGINT, sm_ship_mode_id VARCHAR, sm_type VARCHAR, sm_code VARCHAR, sm_carrier VARCHAR, sm_contract VARCHAR)
TABLE household_demographics (hd_demo_sk BIGINT, hd_income_band_sk BIGINT, hd_buy_potential VARCHAR, hd_dep_count BIGINT, hd_vehicle_count INTEGER)
TABLE income_band (ib_income_band_sk BIGINT, ib_lower_bound BIGINT, ib_upper_bound INTEGER)"""

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {
        "question": "Có bao nhiêu khách hàng?",
        "sql": """SELECT COUNT(DISTINCT c_customer_sk) FROM customer;"""
    },
    {
        "question": "Năm 2002 kênh Store mang về bao nhiêu tiền?",
        "sql": """SELECT SUM(ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002;"""
    },
    {
        "question": "Top 5 sản phẩm có doanh thu cao nhất?",
        "sql": """SELECT i.i_product_name, SUM(ss.ss_net_paid) as revenue FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk GROUP BY i.i_product_name ORDER BY revenue DESC LIMIT 5;"""
    }
]

def build_simple_prompt(question: str, tokenizer, num_few_shot: int = 0, schema_linker=None) -> str:
    """Build prompt matching training data format with optional few-shot examples"""
    system = """You are an expert SQL writer for DuckDB (TPC-DS schema).

CRITICAL RULES (MUST FOLLOW):
1. ONLY use table/column names that EXIST in SCHEMA below
2. DO NOT invent names (e.g., use 'customer' NOT 'customers', 'sr_item_sk' NOT 'return_item_sk')
3. DO NOT use columns that don't exist (e.g., 'c_vehicle_count' is in household_demographics, NOT customer)
4. CHECK exact spellings: 'inv_quantity_on_hand' NOT 'quantity', 'i_item_sk' NOT 'item_sk'
5. Table aliases: customer='c', item='i', store_sales='ss', date_dim='d', store_returns='sr'
6. ALL selected columns must be in GROUP BY (if using aggregation)
7. Use LIMIT (NOT TOP), CURRENT_DATE (NOT getdate())
8. Filter years with d_year from date_dim (NOT year() function)

Output ONLY valid SQL ending with semicolon. No explanations."""
    
    # Select schema: dynamic (via linking) or full
    if schema_linker:
        schema_text = schema_linker.build_dynamic_schema(question, max_tables=5)
    else:
        schema_text = TPCDS_SCHEMA_COMPACT
    
    # Build few-shot section
    few_shot_text = ""
    if num_few_shot > 0:
        examples = FEW_SHOT_EXAMPLES[:num_few_shot]
        few_shot_text = "\n\nEXAMPLES:\n"
        for i, ex in enumerate(examples, 1):
            few_shot_text += f"\nQ{i}: {ex['question']}\nSQL{i}: {ex['sql']}\n"
    
    user = f"SCHEMA:\n{schema_text}{few_shot_text}\n\nQUESTION:\n{question}\n\nSQL:"
    
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True
    )

def benchmark_model(args, tokenizer, model):
    print("\n" + "="*60)
    print("PHASE 2: BENCHMARKING")
    print("="*60)
    
    # Setup DB
    con = setup_db(args.db)
    
    # Initialize schema linker if enabled
    schema_linker = None
    if args.schema_linking and HAS_SCHEMA_LINKING:
        print("Initializing Schema Linker...")
        schema_linker = SchemaLinker()
        print("Schema linking ENABLED (dynamic schema)")
    else:
        print("Schema linking DISABLED (full schema)")
    
    # Load test data (use easy set if --easy flag)
    test_path = args.test_data
    if args.easy:
        test_path = "research_pipeline/datasets/test_easy.csv"
        print(f"Using EASY test set: {test_path}")
    
    test_df = pd.read_csv(test_path)
    test_df = test_df.dropna(subset=["Transcription", "SQL Ground Truth"])
    if args.max_test_samples:
        test_df = test_df.head(args.max_test_samples)
    print(f"Test samples: {len(test_df)}")
    print(f"Few-shot examples: {args.few_shot}")
    
    # Run benchmark
    correct = 0
    valid_count = 0
    results = []
    
    model.eval()
    
    for idx, row in test_df.iterrows():
        question = row["Transcription"]
        gt_sql = row["SQL Ground Truth"]
        
        prompt = build_simple_prompt(question, tokenizer, num_few_shot=args.few_shot, schema_linker=schema_linker)
        
        start = time.time()
        gen_sql = generate_sql(prompt, tokenizer, model)
        gen_time = (time.time() - start) * 1000
        
        # Check validity
        valid = gen_sql.strip().upper().startswith(('SELECT', 'WITH'))
        if valid:
            valid_count += 1
        
        # Run both SQLs
        gt_res, gt_err = run_sql(con, gt_sql)
        gen_res, gen_err = run_sql(con, gen_sql) if valid else (None, "INVALID")
        
        # Compare results
        exec_match = False
        if not gt_err and not gen_err and gt_res is not None and gen_res is not None:
            gt_set = set(str(r) for r in gt_res)
            gen_set = set(str(r) for r in gen_res)
            exec_match = gt_set == gen_set
            if exec_match:
                correct += 1
        
        status = "OK" if exec_match else "FAIL"
        print(f"  [{idx}] {status} Valid={valid}, ExecMatch={exec_match}, Time={gen_time:.0f}ms")
        if gen_err and not exec_match:
            print(f"      Error: {gen_err[:80]}")
        
        results.append({
            "id": idx,
            "question": question[:50],
            "valid": valid,
            "exec_match": exec_match,
            "gen_time_ms": gen_time
        })
    
    con.close()
    
    # Summary
    total = len(test_df)
    print("\n" + "="*60)
    print(f"BENCHMARK RESULTS")
    print("="*60)
    print(f"Valid SQL: {valid_count}/{total} ({100*valid_count/total:.1f}%)")
    print(f"Exec Match: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Save results
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "valid_sql": valid_count,
            "exec_match": correct,
            "total": total,
            "results": results
        }, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return correct / total if total > 0 else 0

# ========== MAIN ==========
def main():
    args = parse_args()
    
    print("="*60)
    print("FINETUNE AND BENCHMARK PIPELINE")
    print("="*60)
    print(f"Train data: {args.train_data}")
    print(f"Test data: {args.test_data}")
    print(f"Adapter: {args.adapter}")
    print(f"Base model only: {args.base_model}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Skip train: {args.skip_train}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model - either base model or adapter
    if args.base_model:
        # Use base model directly (no adapter)
        print(f"\nLoading BASE MODEL: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Load adapter on top of base model
        peft_config = PeftConfig.from_pretrained(args.adapter)
        base_model_id = peft_config.base_model_name_or_path
        print(f"\nBase model: {base_model_id}")
        print(f"Loading adapter: {args.adapter}")
        
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=not args.skip_train)
    
    # Phase 1: Training
    if not args.skip_train:
        model = train_model(args, tokenizer, model)
    else:
        print("\nSkipping training (--skip-train)")
    
    # Phase 2: Benchmark
    accuracy = benchmark_model(args, tokenizer, model)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Final Accuracy: {accuracy*100:.1f}%")


if __name__ == "__main__":
    main()
