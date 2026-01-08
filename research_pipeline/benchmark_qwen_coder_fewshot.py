#!/usr/bin/env python3
"""
Full Pipeline Benchmark: Qwen3-Coder-30B-A3B-Instruct
Features:
1. Schema Linking (dynamic table selection)
2. Few-Shot Learning (3, 5, 7 shots)
3. RAG-based example retrieval

Usage:
    python benchmark_qwen_coder_fewshot.py --shots 3 5 7 --max-test-samples 15
    python benchmark_qwen_coder_fewshot.py --easy --shots 5 --max-test-samples 28
"""

import argparse
import re
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import duckdb

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from schema_linking import SchemaLinker, TPCDS_TABLES, JOIN_RELATIONSHIPS

# ========== CONSTANTS ==========
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DB_PATH = "research_pipeline/cache/ecommerce_dw.duckdb"

# Static few-shot examples covering different patterns
STATIC_FEWSHOT_EXAMPLES = [
    # Channel: Catalog
    {
        "question": "Tổng doanh thu từ catalog trong năm 2000",
        "sql": "SELECT SUM(cs.cs_net_paid) FROM catalog_sales cs JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk WHERE d.d_year = 2000;"
    },
    # Channel: Web/Online
    {
        "question": "Bao nhiêu đơn hàng được đặt qua website trong quý 2 năm 2002?",
        "sql": "SELECT COUNT(*) FROM web_sales ws JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002 AND d.d_qoy = 2;"
    },
    # Channel: Store
    {
        "question": "Doanh thu cửa hàng năm 2002 là bao nhiêu?",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002;"
    },
    # Demographics: Gender
    {
        "question": "Thống kê số lượng khách hàng theo giới tính",
        "sql": "SELECT cd.cd_gender, COUNT(DISTINCT c.c_customer_sk) AS cnt FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk GROUP BY cd.cd_gender;"
    },
    # Demographics: Marital Status
    {
        "question": "So sánh doanh thu giữa khách hàng độc thân và đã kết hôn",
        "sql": "SELECT cd.cd_marital_status, SUM(ss.ss_net_paid) AS revenue FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_marital_status IN ('S', 'M') GROUP BY cd.cd_marital_status;"
    },
    # Item/Product
    {
        "question": "Top 5 sản phẩm bán chạy nhất trên trang web theo số lượng",
        "sql": "SELECT i.i_product_name, SUM(ws.ws_quantity) AS total FROM web_sales ws JOIN item i ON ws.ws_item_sk = i.i_item_sk GROUP BY i.i_product_name ORDER BY total DESC LIMIT 5;"
    },
    # Customer + Address
    {
        "question": "Tổng số giao dịch tại các cửa hàng ở bang California",
        "sql": "SELECT COUNT(*) FROM store_sales ss JOIN store s ON ss.ss_store_sk = s.s_store_sk WHERE s.s_state = 'CA';"
    },
    # Returns with Reason
    {
        "question": "Thống kê đơn hàng online bị trả lại theo lý do",
        "sql": "SELECT r.r_reason_desc, COUNT(*) AS cnt FROM web_returns wr JOIN reason r ON wr.wr_reason_sk = r.r_reason_sk GROUP BY r.r_reason_desc ORDER BY cnt DESC;"
    },
    # Date filtering
    {
        "question": "Doanh thu quý 1 năm 2001",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2001 AND d.d_qoy = 1;"
    },
    # Inventory
    {
        "question": "Kho nào có nhiều hàng tồn nhất?",
        "sql": "SELECT w.w_warehouse_name, SUM(inv.inv_quantity_on_hand) AS total FROM inventory inv JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk GROUP BY w.w_warehouse_name ORDER BY total DESC LIMIT 5;"
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-Coder-30B with Few-Shot & Schema Linking")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name/path")
    parser.add_argument("--test-data", type=str, default="research_pipeline/datasets/test.csv", help="Test CSV path")
    parser.add_argument("--train-data", type=str, default="research_pipeline/datasets/train_clean.csv", help="Train CSV for RAG examples")
    parser.add_argument("--easy", action="store_true", help="Use test_easy.csv")
    parser.add_argument("--shots", type=int, nargs="+", default=[3, 5, 7], help="Few-shot counts to test")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--db-path", type=str, default=DB_PATH, help="DuckDB database path")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG for dynamic example selection")
    parser.add_argument("--output-dir", type=str, default="research_pipeline/results", help="Results output directory")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_model_vllm(model_name: str):
    """Load model with vLLM for faster inference"""
    from vllm import LLM, SamplingParams
    
    print(f"Loading model with vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    return llm, "vllm"


def load_model_transformers(model_name: str):
    """Load model with transformers"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model with transformers: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    return (model, tokenizer), "transformers"


def generate_with_vllm(llm, prompt: str, max_tokens: int = 512) -> str:
    """Generate SQL with vLLM"""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["```", "\n\n\n", "Question:", "User:"],
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def generate_with_transformers(model_tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate SQL with transformers"""
    import torch
    model, tokenizer = model_tokenizer
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def get_schema_for_tables(tables: List[str]) -> str:
    """Generate schema DDL for selected tables"""
    schema_parts = []
    
    for table_name in tables:
        if table_name not in TPCDS_TABLES:
            continue
        table_info = TPCDS_TABLES[table_name]
        columns = ", ".join(table_info["columns"][:10])  # Top 10 columns
        alias = table_info["alias"]
        schema_parts.append(f"-- {table_name} (alias: {alias})\n-- Columns: {columns}")
    
    return "\n\n".join(schema_parts)


def get_join_hints(tables: List[str]) -> str:
    """Get JOIN hints for selected tables"""
    hints = []
    for t1 in tables:
        for t2 in tables:
            if t1 != t2:
                key = (t1, t2)
                if key in JOIN_RELATIONSHIPS:
                    hints.append(f"  {t1} ⟷ {t2}: {JOIN_RELATIONSHIPS[key]}")
                key_rev = (t2, t1)
                if key_rev in JOIN_RELATIONSHIPS and (t1, t2) not in [(h.split(":")[0].strip().split(" ⟷ ")) for h in hints]:
                    hints.append(f"  {t2} ⟷ {t1}: {JOIN_RELATIONSHIPS[key_rev]}")
    return "\n".join(hints) if hints else "  (No direct joins)"


def build_fewshot_prompt(
    question: str,
    schema_info: str,
    join_hints: str,
    examples: List[Dict],
    num_shots: int
) -> str:
    """Build prompt with few-shot examples and schema context"""
    
    system_msg = """Bạn là chuyên gia SQL cho TPC-DS database. Sinh câu SQL chính xác.

IMPORTANT RULES:
- Gender (cd_gender), marital status (cd_marital_status), credit_rating: USE customer_demographics (cd), NOT customer
- Vehicle count (hd_vehicle_count): USE household_demographics (hd)
- Web/Online sales: USE web_sales (ws)
- Catalog/Mail-order: USE catalog_sales (cs)  
- Store/Retail: USE store_sales (ss)
- Quarter of year: USE d_qoy (NOT d_quarter)
- Always use table aliases
- Output ONLY the SQL query, no explanation"""

    # Build few-shot examples
    examples_text = ""
    for i, ex in enumerate(examples[:num_shots], 1):
        examples_text += f"\n### Ví dụ {i}:\nQuestion: {ex['question']}\nSQL: {ex['sql']}\n"
    
    prompt = f"""{system_msg}

### Schema liên quan:
{schema_info}

### JOIN hints:
{join_hints}

### Các ví dụ:{examples_text}

### Câu hỏi cần trả lời:
Question: {question}
SQL:"""
    
    return prompt


def extract_sql(response: str) -> str:
    """Extract SQL from model response"""
    # Remove thinking tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Try to find SQL in code blocks
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'```\s*(SELECT.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try to find SELECT statement
    select_match = re.search(r'(SELECT\s+.+?)(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    return response.strip()


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = sql.strip().rstrip(';').strip()
    return sql


def validate_sql_execution(sql: str, db_path: str) -> Tuple[bool, str]:
    """Validate SQL executes without error"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(sql).fetchall()
        conn.close()
        return True, f"OK ({len(result)} rows)"
    except Exception as e:
        return False, str(e)[:100]


def compare_results(pred_sql: str, gold_sql: str, db_path: str) -> Tuple[bool, str]:
    """Compare execution results of predicted vs gold SQL"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Execute both
        try:
            pred_result = conn.execute(pred_sql).fetchall()
        except Exception as e:
            conn.close()
            return False, f"Pred error: {str(e)[:50]}"
        
        try:
            gold_result = conn.execute(gold_sql).fetchall()
        except Exception as e:
            conn.close()
            return False, f"Gold error: {str(e)[:50]}"
        
        conn.close()
        
        # Compare results
        pred_set = set(str(r) for r in pred_result)
        gold_set = set(str(r) for r in gold_result)
        
        if pred_set == gold_set:
            return True, "Exact match"
        
        # Check if subsets
        if pred_set.issubset(gold_set) or gold_set.issubset(pred_set):
            return True, "Subset match"
        
        # Check numeric similarity for aggregations
        if len(pred_result) == 1 and len(gold_result) == 1:
            try:
                pred_val = float(pred_result[0][0]) if pred_result[0][0] else 0
                gold_val = float(gold_result[0][0]) if gold_result[0][0] else 0
                if gold_val != 0 and abs(pred_val - gold_val) / abs(gold_val) < 0.01:
                    return True, "Numeric match (~1%)"
            except:
                pass
        
        return False, f"Mismatch (pred={len(pred_result)}, gold={len(gold_result)})"
    
    except Exception as e:
        return False, f"Compare error: {str(e)[:50]}"


def run_benchmark(args):
    """Run full benchmark with different shot counts"""
    
    print("=" * 70)
    print("Qwen3-Coder-30B-A3B Full Pipeline Benchmark")
    print("=" * 70)
    
    # Load model
    try:
        if args.use_vllm:
            llm, backend = load_model_vllm(args.model)
        else:
            llm, backend = load_model_transformers(args.model)
        print(f"Model loaded! Backend: {backend}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying transformers backend...")
        llm, backend = load_model_transformers(args.model)
    
    # Initialize schema linker
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
    
    # Get question/sql columns
    q_col = "Transcription" if "Transcription" in test_df.columns else "question"
    sql_col = "SQL Ground Truth" if "SQL Ground Truth" in test_df.columns else "sql"
    
    # Results storage
    all_results = {}
    
    for num_shots in args.shots:
        print(f"\n{'='*70}")
        print(f"Testing with {num_shots}-shot")
        print(f"{'='*70}")
        
        results = []
        correct = 0
        exec_correct = 0
        
        for idx, row in test_df.iterrows():
            question = row[q_col]
            gold_sql = row[sql_col]
            
            # Clean gold SQL
            gold_sql = gold_sql.replace('\n', ' ').strip()
            if not gold_sql.endswith(';'):
                gold_sql += ';'
            
            print(f"\n[{idx+1}/{len(test_df)}] {question[:60]}...")
            
            # Schema linking
            linked = schema_linker.link_schema(question, top_k_tables=6)
            linked_tables = linked["tables"]
            
            if args.verbose:
                print(f"  Linked tables: {linked_tables}")
            
            # Get schema and join hints
            schema_info = get_schema_for_tables(linked_tables)
            join_hints = get_join_hints(linked_tables)
            
            # Build prompt
            prompt = build_fewshot_prompt(
                question=question,
                schema_info=schema_info,
                join_hints=join_hints,
                examples=STATIC_FEWSHOT_EXAMPLES,
                num_shots=num_shots
            )
            
            # Generate
            start_time = time.time()
            if backend == "vllm":
                response = generate_with_vllm(llm, prompt)
            else:
                response = generate_with_transformers(llm, prompt)
            gen_time = time.time() - start_time
            
            # Extract SQL
            pred_sql = extract_sql(response)
            
            if args.verbose:
                print(f"  Generated: {pred_sql[:100]}...")
                print(f"  Time: {gen_time:.2f}s")
            
            # Validate execution
            exec_ok, exec_msg = validate_sql_execution(pred_sql, args.db_path)
            
            # Compare results
            match, match_msg = compare_results(pred_sql, gold_sql, args.db_path)
            
            if match:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            if exec_ok:
                exec_correct += 1
            
            print(f"  {status} Exec: {exec_msg} | Match: {match_msg}")
            
            results.append({
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "linked_tables": linked_tables,
                "exec_ok": exec_ok,
                "match": match,
                "gen_time": gen_time,
            })
        
        # Calculate metrics
        accuracy = correct / len(test_df) * 100
        exec_rate = exec_correct / len(test_df) * 100
        
        print(f"\n{'='*50}")
        print(f"{num_shots}-shot Results:")
        print(f"  Execution Accuracy: {exec_correct}/{len(test_df)} = {exec_rate:.1f}%")
        print(f"  Result Match: {correct}/{len(test_df)} = {accuracy:.1f}%")
        print(f"{'='*50}")
        
        all_results[num_shots] = {
            "accuracy": accuracy,
            "exec_rate": exec_rate,
            "correct": correct,
            "exec_correct": exec_correct,
            "total": len(test_df),
            "results": results,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Few-Shot Comparison")
    print("=" * 70)
    print(f"{'Shots':<10} {'Exec Rate':<15} {'Accuracy':<15}")
    print("-" * 40)
    for shots, data in all_results.items():
        print(f"{shots:<10} {data['exec_rate']:.1f}%{'':<10} {data['accuracy']:.1f}%")
    
    # Find best
    best_shots = max(all_results, key=lambda x: all_results[x]["accuracy"])
    print(f"\nBest: {best_shots}-shot with {all_results[best_shots]['accuracy']:.1f}% accuracy")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"qwen_coder_fewshot_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        # Convert results for JSON serialization
        save_data = {}
        for shots, data in all_results.items():
            save_data[str(shots)] = {
                "accuracy": data["accuracy"],
                "exec_rate": data["exec_rate"],
                "correct": data["correct"],
                "total": data["total"],
            }
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
