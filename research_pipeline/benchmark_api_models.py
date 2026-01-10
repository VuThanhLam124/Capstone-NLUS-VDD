#!/usr/bin/env python3
"""
Benchmark API Models for Text-to-SQL
Supports: OpenAI (GPT-4o, GPT-4-turbo), Google Gemini (1.5 Pro, 1.5 Flash)

Usage:
    # OpenAI
    export OPENAI_API_KEY="sk-..."
    python benchmark_api_models.py --model gpt-4o --shots 7 --max-samples 50
    
    # Gemini
    export GOOGLE_API_KEY="..."
    python benchmark_api_models.py --model gemini-1.5-pro --shots 7 --max-samples 50
"""

import argparse
import os
import re
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import duckdb

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from schema_linking import SchemaLinker, TPCDS_TABLES
from prompt_assets import load_few_shot_examples

# Try importing API clients
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Run: pip install openai")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  Gemini not installed. Run: pip install google-generativeai")

# ========== CONSTANTS ==========
DB_PATH = "research_pipeline/cache/ecommerce_dw.duckdb"

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {"provider": "openai", "model_name": "gpt-4o", "max_tokens": 1000},
    "gpt-4o-mini": {"provider": "openai", "model_name": "gpt-4o-mini", "max_tokens": 1000},
    "gpt-4-turbo": {"provider": "openai", "model_name": "gpt-4-turbo-preview", "max_tokens": 1000},
    
    # Gemini models
    "gemini-1.5-pro": {"provider": "gemini", "model_name": "gemini-1.5-pro", "max_tokens": 1000},
    "gemini-1.5-flash": {"provider": "gemini", "model_name": "gemini-1.5-flash", "max_tokens": 1000},
}

# Static few-shot examples (same as Qwen benchmark)
STATIC_FEWSHOT_EXAMPLES = load_few_shot_examples("benchmark")


class APIModelBenchmark:
    """Benchmark API models for Text-to-SQL"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name)
        if not self.config:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        
        self.provider = self.config["provider"]
        self.schema_linker = SchemaLinker()
        
        # Initialize API client
        if self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("OpenAI not installed. Run: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI(api_key=api_key)
            
        elif self.provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError("Gemini not installed. Run: pip install google-generativeai")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.config["model_name"])
    
    def generate_sql(self, question: str, schema_info: str, join_hints: str, 
                     examples: List[Dict], num_shots: int) -> str:
        """Generate SQL using API"""
        prompt = self._build_prompt(question, schema_info, join_hints, examples, num_shots)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config["model_name"],
                    messages=[
                        {"role": "system", "content": "You are an expert SQL assistant for TPC-DS database."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config["max_tokens"],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "gemini":
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.config["max_tokens"],
                        temperature=0.0
                    )
                )
                return response.text.strip()
        
        except Exception as e:
            print(f"❌ API Error: {e}")
            return ""
    
    def _build_prompt(self, question: str, schema_info: str, join_hints: str,
                     examples: List[Dict], num_shots: int) -> str:
        """Build prompt with few-shot examples"""
        
        system_msg = """Bạn là chuyên gia SQL cho TPC-DS database. Sinh câu SQL chính xác.

=== CRITICAL RULES ===
1. KHÔNG thêm filter (WHERE) nếu câu hỏi KHÔNG yêu cầu
2. "bán chạy nhất" = SUM(quantity), KHÔNG phải SUM(sales_price)
3. "trả lại hàng" → mặc định dùng store_returns (sr)
4. "từ X trở lên" = >= X
5. Chỉ SELECT các columns cần thiết

=== COLUMN MAPPINGS ===
- Email: c.c_email_address (NOT c_email)
- Gender: cd.cd_gender (customer_demographics, NOT customer)
- Marital: cd.cd_marital_status (customer_demographics)
- Vehicle: hd.hd_vehicle_count (household_demographics)
- Tax: ss.ss_ext_tax (NOT ss_tax)
- Quarter: d.d_qoy (NOT d_quarter)

=== CHANNEL RULES ===
- "cửa hàng" → store_sales (ss)
- "online/web" → web_sales (ws)
- "catalog" → catalog_sales (cs)

Output ONLY the SQL query, no explanation."""
        
        # Build examples
        examples_text = ""
        for i, ex in enumerate(examples[:num_shots], 1):
            examples_text += f"\n### Example {i}:\nQuestion: {ex['question']}\nSQL: {ex['sql']}\n"
        
        prompt = f"""{system_msg}

### Relevant Schema:
{schema_info}

### JOIN Hints:
{join_hints}

### Examples:{examples_text}

### Question to answer:
Question: {question}
SQL:"""
        
        return prompt


def get_schema_for_tables(tables: List[str]) -> str:
    """Generate schema DDL for selected tables"""
    schema_parts = []
    for table_name in tables:
        if table_name not in TPCDS_TABLES:
            continue
        table_info = TPCDS_TABLES[table_name]
        columns = ", ".join(table_info["columns"])
        alias = table_info["alias"]
        schema_parts.append(f"-- {table_name} (alias: {alias})\n-- Columns: {columns}")
    return "\n\n".join(schema_parts)


def extract_sql(response: str) -> str:
    """Extract SQL from model response"""
    # Remove markdown code blocks
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'```\s*(SELECT.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try to find SELECT statement
    select_match = re.search(r'(SELECT\s+.+?)(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return response.strip()


def execute_sql(sql: str, db_path: str) -> Tuple[bool, str, List]:
    """Execute SQL and return (success, error_msg, results)"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(sql).fetchall()
        conn.close()
        return True, "", result
    except Exception as e:
        return False, str(e), []


def normalize_results(results: List) -> set:
    """Normalize results for comparison"""
    normalized = set()
    for row in results:
        normalized_row = tuple(
            round(float(val), 2) if isinstance(val, (int, float)) else val
            for val in row
        )
        normalized.add(normalized_row)
    return normalized


def run_benchmark(args):
    """Run benchmark on test dataset"""
    
    print("=" * 80)
    print(f"API MODEL BENCHMARK: {args.model}")
    print("=" * 80)
    print(f"Dataset: {'test_easy.csv' if args.easy else 'test.csv'}")
    print(f"Few-shot: {args.shots}")
    print(f"Max samples: {args.max_samples}")
    print(f"Provider: {MODEL_CONFIGS[args.model]['provider']}")
    print("=" * 80)
    
    # Load dataset
    dataset_file = "research_pipeline/datasets/test_easy.csv" if args.easy else "research_pipeline/datasets/test.csv"
    df = pd.read_csv(dataset_file)
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    print(f"\n📊 Loaded {len(df)} test samples\n")
    
    # Initialize benchmark
    benchmark = APIModelBenchmark(args.model)
    
    results = []
    total_time = 0
    
    for idx, row in df.iterrows():
        question = row['Transcription']
        ground_truth_sql = row['SQL Ground Truth']
        
        print(f"\n[{idx+1}/{len(df)}] {question[:60]}...")
        
        # Schema linking
        linking_result = benchmark.schema_linker.link_schema(question, top_k_tables=5)
        linked_tables = linking_result["tables"]
        schema_info = get_schema_for_tables(linked_tables)
        join_hints = "\n".join(linking_result["joins"][:3]) or "(No direct joins)"
        
        # Generate SQL
        start_time = time.time()
        response = benchmark.generate_sql(
            question, schema_info, join_hints, 
            STATIC_FEWSHOT_EXAMPLES, args.shots
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        
        predicted_sql = extract_sql(response)
        
        # Execute both SQLs
        gt_success, gt_error, gt_results = execute_sql(ground_truth_sql, DB_PATH)
        pred_success, pred_error, pred_results = execute_sql(predicted_sql, DB_PATH)
        
        # Compare results
        execution_match = False
        result_match = False
        
        if gt_success and pred_success:
            execution_match = True
            gt_norm = normalize_results(gt_results)
            pred_norm = normalize_results(pred_results)
            result_match = (gt_norm == pred_norm)
        
        status = "✅" if result_match else "❌"
        print(f"{status} Exec: {pred_success}, Match: {result_match}, Time: {elapsed:.1f}s")
        
        if args.verbose and not result_match:
            print(f"   GT SQL:   {ground_truth_sql[:100]}...")
            print(f"   Pred SQL: {predicted_sql[:100]}...")
            if not pred_success:
                print(f"   Error: {pred_error[:100]}...")
        
        results.append({
            "question": question,
            "ground_truth_sql": ground_truth_sql,
            "predicted_sql": predicted_sql,
            "execution_success": pred_success,
            "result_match": result_match,
            "error": pred_error if not pred_success else "",
            "time_seconds": elapsed
        })
        
        # Rate limiting
        time.sleep(0.5)  # Avoid hitting API rate limits
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_{args.model.replace('-', '_')}_{args.shots}shot_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    total = len(results)
    exec_success = sum(1 for r in results if r["execution_success"])
    match_success = sum(1 for r in results if r["result_match"])
    
    print(f"Total samples:         {total}")
    print(f"Execution success:     {exec_success}/{total} ({exec_success/total*100:.1f}%)")
    print(f"Result match:          {match_success}/{total} ({match_success/total*100:.1f}%)")
    print(f"Total time:            {total_time:.1f}s")
    print(f"Avg time/query:        {total_time/total:.1f}s")
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark API Models for Text-to-SQL")
    parser.add_argument("--model", type=str, required=True, 
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to benchmark")
    parser.add_argument("--shots", type=int, default=7,
                       help="Number of few-shot examples (default: 7)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max test samples (default: all)")
    parser.add_argument("--easy", action="store_true",
                       help="Use test_easy.csv instead of test.csv")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed errors")
    
    args = parser.parse_args()
    
    # Check API keys
    if MODEL_CONFIGS[args.model]["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    if MODEL_CONFIGS[args.model]["provider"] == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("   Set it with: export GOOGLE_API_KEY='...'")
        sys.exit(1)
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
