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
    text = re.sub(r'^```sql\s*', '', text.strip())
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'```$', '', text)
    
    if ';' in text:
        text = text[:text.index(';')+1]
    
    return text.strip()

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
    return extract_sql(tokenizer.decode(gen_ids, skip_special_tokens=True))

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

def build_simple_prompt(question: str, tokenizer) -> str:
    system = f"""You are an expert SQL writer for DuckDB using the TPC-DS e-commerce schema.
Output ONLY valid SQL ending with semicolon. Use ONLY the tables and columns listed below.

{TPCDS_SCHEMA}

IMPORTANT RULES:
1. Use ONLY exact table and column names from the schema above
2. For year filtering, JOIN with date_dim and use d_year (e.g., WHERE d.d_year = 2000)
3. Do NOT use year() function on date_sk columns - they are integers, not dates
4. Table aliases: use the abbreviation in parentheses (e.g., store_sales AS ss)
5. Always JOIN dimension tables to get readable data (e.g., item for product names)"""

    user = f"QUESTION: {question}\n\nSQL:"
    
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
    
    # Load test data
    test_df = pd.read_csv(args.test_data)
    test_df = test_df.dropna(subset=["Transcription", "SQL Ground Truth"])
    if args.max_test_samples:
        test_df = test_df.head(args.max_test_samples)
    print(f"Test samples: {len(test_df)}")
    
    # Run benchmark
    correct = 0
    valid_count = 0
    results = []
    
    model.eval()
    
    for idx, row in test_df.iterrows():
        question = row["Transcription"]
        gt_sql = row["SQL Ground Truth"]
        
        prompt = build_simple_prompt(question, tokenizer)
        
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
    results_path = Path(args.output) / "benchmark_results.json"
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
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Skip train: {args.skip_train}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load base model and adapter
    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_model_id = peft_config.base_model_name_or_path
    print(f"\nBase model: {base_model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
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
