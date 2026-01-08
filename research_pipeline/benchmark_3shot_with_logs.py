#!/usr/bin/env python3
"""
3-Shot Benchmark with Detailed Error Logging
Usage:
    python benchmark_3shot_with_logs.py --easy
    python benchmark_3shot_with_logs.py --max-test-samples 50
"""

import argparse
import re
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import duckdb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from schema_linking import SchemaLinker, TPCDS_TABLES

# ========== CONSTANTS ==========
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DB_PATH = "research_pipeline/cache/ecommerce_dw.duckdb"

# 3-Shot Examples
EXAMPLES = [
    {
        "question": "Tổng doanh thu từ catalog trong năm 2000",
        "sql": "SELECT SUM(cs.cs_net_paid) FROM catalog_sales cs JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk WHERE d.d_year = 2000;"
    },
    {
        "question": "Thống kê số lượng khách hàng theo giới tính",
        "sql": "SELECT cd.cd_gender, COUNT(DISTINCT c.c_customer_sk) AS cnt FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk GROUP BY cd.cd_gender;"
    },
    {
        "question": "Top 5 sản phẩm bán chạy nhất trên trang web theo số lượng",
        "sql": "SELECT i.i_product_name, SUM(ws.ws_quantity) AS total FROM web_sales ws JOIN item i ON ws.ws_item_sk = i.i_item_sk GROUP BY i.i_product_name ORDER BY total DESC LIMIT 5;"
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--test-data", type=str, default="research_pipeline/datasets/test.csv")
    parser.add_argument("--easy", action="store_true", help="Use test_easy.csv")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    parser.add_argument("--output-dir", type=str, default="research_pipeline/results")
    return parser.parse_args()


def load_model(model_name: str):
    """Load model and tokenizer"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    return model, tokenizer


def get_schema_for_tables(tables: List[str]) -> str:
    """Generate schema DDL for selected tables"""
    schema_parts = []
    for table_name in tables[:6]:  # Top 6 tables
        if table_name not in TPCDS_TABLES:
            continue
        table_info = TPCDS_TABLES[table_name]
        columns = ", ".join(table_info["columns"][:8])
        alias = table_info["alias"]
        schema_parts.append(f"-- {table_name} ({alias}): {columns}")
    return "\n".join(schema_parts)


def build_prompt(question: str, schema_info: str) -> str:
    """Build 3-shot prompt"""
    system_msg = """Bạn là chuyên gia SQL cho TPC-DS. Sinh câu SQL chính xác.

CRITICAL RULES:
- Gender (cd_gender), marital_status, credit_rating → customer_demographics (cd), NOT customer
- Vehicle count (hd_vehicle_count) → household_demographics (hd)
- Web/Online sales → web_sales (ws)
- Catalog sales → catalog_sales (cs)  
- Store/Retail → store_sales (ss)
- Quarter → d_qoy (NOT d_quarter)
- Output ONLY SQL, no explanation"""

    examples_text = ""
    for i, ex in enumerate(EXAMPLES, 1):
        examples_text += f"\n### Ví dụ {i}:\nQuestion: {ex['question']}\nSQL: {ex['sql']}\n"
    
    prompt = f"""{system_msg}

### Schema:
{schema_info}

### Ví dụ:{examples_text}

### Câu hỏi:
Question: {question}
SQL:"""
    
    return prompt


def generate_sql(model, tokenizer, prompt: str) -> str:
    """Generate SQL with model"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_sql(response: str) -> str:
    """Extract SQL from response"""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'```\s*(SELECT.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    select_match = re.search(r'(SELECT\s+.+?)(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    return response.strip()


def validate_sql(sql: str, db_path: str) -> Tuple[bool, str, list]:
    """Validate SQL and return results"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(sql).fetchall()
        conn.close()
        return True, f"OK ({len(result)} rows)", result
    except Exception as e:
        return False, str(e)[:200], []


def compare_results(pred_result: list, gold_result: list) -> Tuple[bool, str]:
    """Compare query results"""
    pred_set = set(str(r) for r in pred_result)
    gold_set = set(str(r) for r in gold_result)
    
    if pred_set == gold_set:
        return True, "Exact match"
    
    if pred_set.issubset(gold_set) or gold_set.issubset(pred_set):
        return True, "Subset match"
    
    # Numeric similarity
    if len(pred_result) == 1 and len(gold_result) == 1:
        try:
            pred_val = float(pred_result[0][0]) if pred_result[0][0] else 0
            gold_val = float(gold_result[0][0]) if gold_result[0][0] else 0
            if gold_val != 0 and abs(pred_val - gold_val) / abs(gold_val) < 0.01:
                return True, "Numeric ~1%"
        except:
            pass
    
    return False, f"Mismatch (pred={len(pred_result)}, gold={len(gold_result)})"


def analyze_error(pred_sql: str, gold_sql: str, question: str) -> Dict:
    """Analyze common error patterns"""
    errors = {
        "channel_mistake": False,
        "demographics_mistake": False,
        "table_mistake": False,
        "column_mistake": False,
        "join_mistake": False,
    }
    
    pred_lower = pred_sql.lower()
    gold_lower = gold_sql.lower()
    
    # Channel mistakes
    if "catalog" in question.lower():
        if "catalog_sales" in gold_lower and "catalog_sales" not in pred_lower:
            errors["channel_mistake"] = "Expected catalog_sales"
    if "online" in question.lower() or "website" in question.lower() or "web" in question.lower():
        if "web_sales" in gold_lower and "web_sales" not in pred_lower:
            errors["channel_mistake"] = "Expected web_sales"
    if "cửa hàng" in question.lower() or "store" in question.lower():
        if "store_sales" in gold_lower and "store_sales" not in pred_lower:
            errors["channel_mistake"] = "Expected store_sales"
    
    # Demographics mistakes
    if any(kw in question.lower() for kw in ["giới tính", "gender", "nam", "nữ", "male", "female"]):
        if "customer_demographics" in gold_lower and "customer_demographics" not in pred_lower:
            errors["demographics_mistake"] = "Missing customer_demographics for gender"
    
    if any(kw in question.lower() for kw in ["xe", "vehicle"]):
        if "household_demographics" in gold_lower and "household_demographics" not in pred_lower:
            errors["demographics_mistake"] = "Missing household_demographics for vehicle"
    
    # Table presence check
    gold_tables = set(re.findall(r'\b(?:from|join)\s+(\w+)', gold_lower))
    pred_tables = set(re.findall(r'\b(?:from|join)\s+(\w+)', pred_lower))
    
    missing_tables = gold_tables - pred_tables
    if missing_tables:
        errors["table_mistake"] = f"Missing tables: {missing_tables}"
    
    # JOIN count
    gold_joins = len(re.findall(r'\bjoin\b', gold_lower))
    pred_joins = len(re.findall(r'\bjoin\b', pred_lower))
    if gold_joins != pred_joins:
        errors["join_mistake"] = f"JOIN count: pred={pred_joins}, gold={gold_joins}"
    
    return errors


def main():
    args = parse_args()
    
    print("=" * 70)
    print("3-Shot Benchmark with Error Analysis")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Load schema linker
    print("\nInitializing Schema Linker...")
    schema_linker = SchemaLinker()
    
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
    
    q_col = "Transcription" if "Transcription" in test_df.columns else "question"
    sql_col = "SQL Ground Truth" if "SQL Ground Truth" in test_df.columns else "sql"
    
    # Results
    results = []
    correct = 0
    exec_correct = 0
    
    # Error statistics
    error_stats = {
        "channel_mistakes": 0,
        "demographics_mistakes": 0,
        "table_mistakes": 0,
        "join_mistakes": 0,
        "execution_errors": 0,
    }
    
    # Detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"benchmark_3shot_log_{timestamp}.txt"
    error_log_file = output_dir / f"error_analysis_{timestamp}.txt"
    
    with open(log_file, "w", encoding="utf-8") as log_f, \
         open(error_log_file, "w", encoding="utf-8") as err_f:
        
        log_f.write("=" * 80 + "\n")
        log_f.write("3-Shot Benchmark Detailed Log\n")
        log_f.write(f"Model: {args.model}\n")
        log_f.write(f"Test: {test_path}\n")
        log_f.write(f"Timestamp: {timestamp}\n")
        log_f.write("=" * 80 + "\n\n")
        
        err_f.write("ERROR ANALYSIS LOG\n")
        err_f.write("=" * 80 + "\n\n")
        
        for idx, row in test_df.iterrows():
            question = row[q_col]
            gold_sql = row[sql_col].replace('\n', ' ').strip()
            if not gold_sql.endswith(';'):
                gold_sql += ';'
            
            print(f"\n[{idx+1}/{len(test_df)}] {question[:60]}...")
            log_f.write(f"\n{'='*80}\n")
            log_f.write(f"[{idx+1}/{len(test_df)}]\n")
            log_f.write(f"Question: {question}\n")
            log_f.write(f"Gold SQL: {gold_sql}\n\n")
            
            # Schema linking
            linked = schema_linker.link_schema(question, top_k_tables=6)
            schema_info = get_schema_for_tables(linked["tables"])
            
            log_f.write(f"Linked Tables: {linked['tables']}\n\n")
            
            # Build prompt & generate
            prompt = build_prompt(question, schema_info)
            start_time = time.time()
            response = generate_sql(model, tokenizer, prompt)
            gen_time = time.time() - start_time
            
            pred_sql = extract_sql(response)
            
            log_f.write(f"Predicted SQL: {pred_sql}\n")
            log_f.write(f"Generation time: {gen_time:.2f}s\n\n")
            
            # Validate execution
            exec_ok, exec_msg, pred_result = validate_sql(pred_sql, args.db_path)
            _, _, gold_result = validate_sql(gold_sql, args.db_path)
            
            if exec_ok:
                exec_correct += 1
            else:
                error_stats["execution_errors"] += 1
                log_f.write(f"❌ EXECUTION ERROR: {exec_msg}\n")
            
            # Compare results
            if exec_ok and gold_result:
                match, match_msg = compare_results(pred_result, gold_result)
            else:
                match = False
                match_msg = "Cannot compare (execution failed)"
            
            if match:
                correct += 1
                status = "✓"
                log_f.write(f"✓ CORRECT\n")
            else:
                status = "✗"
                log_f.write(f"✗ WRONG: {match_msg}\n")
                
                # Analyze error
                error_analysis = analyze_error(pred_sql, gold_sql, question)
                
                err_f.write(f"\n{'='*80}\n")
                err_f.write(f"[{idx+1}] {question}\n")
                err_f.write(f"\nGold SQL:\n{gold_sql}\n")
                err_f.write(f"\nPred SQL:\n{pred_sql}\n")
                err_f.write(f"\nError Analysis:\n")
                
                for error_type, error_msg in error_analysis.items():
                    if error_msg:
                        err_f.write(f"  - {error_type}: {error_msg}\n")
                        if "channel" in error_type:
                            error_stats["channel_mistakes"] += 1
                        elif "demographics" in error_type:
                            error_stats["demographics_mistakes"] += 1
                        elif "table" in error_type:
                            error_stats["table_mistakes"] += 1
                        elif "join" in error_type:
                            error_stats["join_mistakes"] += 1
                
                err_f.write(f"\nExecution: {exec_msg}\n")
                err_f.write(f"Match: {match_msg}\n")
            
            print(f"  {status} Exec: {exec_msg[:50]} | Match: {match_msg}")
            
            results.append({
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "exec_ok": exec_ok,
                "match": match,
                "gen_time": gen_time,
            })
        
        # Summary
        accuracy = correct / len(test_df) * 100
        exec_rate = exec_correct / len(test_df) * 100
        
        summary = f"""
{'='*70}
SUMMARY
{'='*70}
Total samples: {len(test_df)}
Execution success: {exec_correct}/{len(test_df)} = {exec_rate:.1f}%
Result match: {correct}/{len(test_df)} = {accuracy:.1f}%

ERROR STATISTICS:
- Channel mistakes: {error_stats['channel_mistakes']}
- Demographics mistakes: {error_stats['demographics_mistakes']}
- Table mistakes: {error_stats['table_mistakes']}
- JOIN mistakes: {error_stats['join_mistakes']}
- Execution errors: {error_stats['execution_errors']}
{'='*70}
"""
        
        print(summary)
        log_f.write(summary)
        err_f.write(summary)
    
    print(f"\n✓ Logs saved:")
    print(f"  - {log_file}")
    print(f"  - {error_log_file}")
    
    # Save JSON results
    json_file = output_dir / f"benchmark_3shot_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "accuracy": accuracy,
                "exec_rate": exec_rate,
                "total": len(test_df),
                "correct": correct,
                "exec_correct": exec_correct,
            },
            "error_stats": error_stats,
            "results": results[:10],  # Only save first 10 for JSON
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  - {json_file}")


if __name__ == "__main__":
    main()
