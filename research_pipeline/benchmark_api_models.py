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
import requests  # Cho Vertex AI REST API

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
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  Gemini not installed. Run: pip install google-genai")

# ========== CONSTANTS ==========
DB_PATH = "research_pipeline/cache/ecommerce_dw.duckdb"

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models - GPT-5 series (mới nhất)
    "gpt-5": {"provider": "openai", "model_name": "gpt-5", "max_tokens": 4000},
    "gpt-5-codex": {"provider": "openai", "model_name": "gpt-5-codex", "max_tokens": 4000},
    "gpt-5.1-codex": {"provider": "openai", "model_name": "gpt-5.1-codex", "max_tokens": 4000},
    "gpt-5.2": {"provider": "openai", "model_name": "gpt-5.2", "max_tokens": 4000},
    "gpt-5.2-codex": {"provider": "openai", "model_name": "gpt-5.2-codex", "max_tokens": 4000},
    # OpenAI models - o series (reasoning)
    "o3": {"provider": "openai", "model_name": "o3", "max_tokens": 4000},
    "o3-mini": {"provider": "openai", "model_name": "o3-mini", "max_tokens": 4000},
    "o1": {"provider": "openai", "model_name": "o1", "max_tokens": 4000},
    "o1-mini": {"provider": "openai", "model_name": "o1-mini", "max_tokens": 4000},
    # OpenAI models - GPT-4 series
    "gpt-4o": {"provider": "openai", "model_name": "gpt-4o", "max_tokens": 1000},
    "gpt-4o-mini": {"provider": "openai", "model_name": "gpt-4o-mini", "max_tokens": 1000},
    "gpt-4-turbo": {"provider": "openai", "model_name": "gpt-4-turbo-preview", "max_tokens": 1000},
    
    # Gemini models (Vertex AI) - tăng max_tokens lên 2000
    "gemini-3-pro": {"provider": "gemini", "model_name": "gemini-3-pro-preview", "max_tokens": 2000},
    "gemini-3-flash": {"provider": "gemini", "model_name": "gemini-3-flash-preview", "max_tokens": 2000},
    "gemini-2.5-pro": {"provider": "gemini", "model_name": "gemini-2.5-pro", "max_tokens": 2000},
    "gemini-2.5-flash": {"provider": "gemini", "model_name": "gemini-2.5-flash", "max_tokens": 2000},
    "gemini-2.0-flash": {"provider": "gemini", "model_name": "gemini-2.0-flash", "max_tokens": 2000},
    "gemini-1.5-pro": {"provider": "gemini", "model_name": "gemini-1.5-pro", "max_tokens": 2000},
    "gemini-1.5-flash": {"provider": "gemini", "model_name": "gemini-1.5-flash", "max_tokens": 2000},
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
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            self.api_key = api_key
            # Xác định loại API key: Vertex AI (AQ.xxx) hay Generative AI
            self.use_vertex_rest = api_key.startswith("AQ.")
            if not self.use_vertex_rest:
                if not HAS_GEMINI:
                    raise ImportError("Gemini not installed. Run: pip install google-genai")
                self.client = genai.Client(api_key=api_key)
    
    def generate_sql(self, question: str, schema_info: str, join_hints: str, 
                     examples: List[Dict], num_shots: int) -> str:
        """Generate SQL using API"""
        prompt = self._build_prompt(question, schema_info, join_hints, examples, num_shots)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    model_name = self.config["model_name"]
                    # o1, o3 models không hỗ trợ system message và dùng max_completion_tokens
                    is_reasoning_model = model_name.startswith("o1") or model_name.startswith("o3")
                    
                    if is_reasoning_model:
                        # o1 models: không có system message, dùng max_completion_tokens
                        response = self.client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "user", "content": f"You are an expert SQL assistant for TPC-DS database.\n\n{prompt}"}
                            ],
                            max_completion_tokens=self.config["max_tokens"]
                        )
                    else:
                        # GPT-4o và các model khác
                        response = self.client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert SQL assistant for TPC-DS database."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=self.config["max_tokens"],
                            temperature=0.0
                        )
                    return response.choices[0].message.content.strip()
                
                elif self.provider == "gemini":
                    if self.use_vertex_rest:
                        # Gọi Vertex AI REST API trực tiếp
                        return self._call_vertex_rest_api(prompt)
                    else:
                        # Gọi qua SDK
                        response = self.client.models.generate_content(
                            model=self.config["model_name"],
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=self.config["max_tokens"],
                                temperature=0.0
                            )
                        )
                        return response.text.strip()
            
            except Exception as e:
                error_str = str(e)
                # Retry nếu gặp rate limiting (429)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    print(f"⏳ Rate limited, waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ API Error: {e}")
                    return ""
        
        print(f"❌ Max retries exceeded")
        return ""
    
    def _call_vertex_rest_api(self, prompt: str) -> str:
        """Gọi Vertex AI REST API trực tiếp với API key"""
        model_name = self.config["model_name"]
        # Endpoint format for Generative Language API (using API key)
        # Note: 'gemini-2.5-pro' requires the v1beta endpoint sometimes, but let's try v1 first or v1beta appropriately if needed. 
        # Actually, for API key (AQ...), we should use generativelanguage.googleapis.com usually, but the user's curl worked with aiplatform and API key?
        # User output showed: "Publisher Model ... not found". 
        # Let's keep aiplatform if that's what worked for gemini-3-flash, but handle errors better.
        
        url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.config["max_tokens"],
                "temperature": 0.0
            }
        }
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120  # Increased timeout
        )
        
        if response.status_code != 200:
            # Try to extract error message
            try:
                error_body = response.json()
                error_msg = json.dumps(error_body, indent=2)
            except:
                error_msg = response.text[:500]
            raise Exception(f"{response.status_code}: {error_msg}")
        
        result = response.json()
        
        # Lấy text từ response
        try:
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                
                # Check for finishReason
                finish_reason = candidate.get("finishReason")
                if finish_reason and finish_reason != "STOP":
                    print(f"⚠️ Finish Reason: {finish_reason}")
                
                if "content" in candidate and "parts" in candidate["content"]:
                    return candidate["content"]["parts"][0].get("text", "").strip()
            
            # Fallback debug
            print(f"⚠️ Empty/Unexpected response structure: {json.dumps(result)[:200]}...")
            return ""
            
        except Exception as parse_error:
            print(f"❌ Parse Error: {parse_error} - Raw: {json.dumps(result)[:200]}")
            return ""
    
    def _build_prompt(self, question: str, schema_info: str, join_hints: str,
                     examples: List[Dict], num_shots: int) -> str:
        """Build prompt with few-shot examples - đồng bộ với finetune_qwen_coder.py"""
        from business_rules import load_business_rules
        
        rules = load_business_rules()
        system_rules = "\n".join([
            "Bạn là chuyên gia SQL cho TPC-DS database. Sinh câu SQL chính xác.",
            "",
            rules,
        ])

        prompt_parts = [
            system_rules,
            "",
            "DATABASE SCHEMA:",
            schema_info,
        ]
        
        # Add JOIN hints
        if join_hints and join_hints != "(No direct joins)":
            prompt_parts.append("\nJOIN HINTS:")
            prompt_parts.append(join_hints)
        
        # Add few-shot examples
        if num_shots > 0 and examples:
            prompt_parts.append("\nEXAMPLES:")
            for ex in examples[:num_shots]:
                prompt_parts.append(f"\nQ: {ex['question']}")
                prompt_parts.append(f"SQL:\n{ex['sql']}")
        
        prompt_parts.extend([
            "",
            f"QUESTION: {question}",
            "",
            "SQL:"
        ])
        
        return "\n".join(prompt_parts)


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
    if args.dataset:
        dataset_file = args.dataset
    else:
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
        
        # Schema linking - sử dụng build_dynamic_schema như finetune_qwen_coder.py
        schema_info = benchmark.schema_linker.build_dynamic_schema(question, max_tables=5)
        linking_result = benchmark.schema_linker.link_schema(question, top_k_tables=5)
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
        time.sleep(3)  # Delay 3s cho Tier 1 rate limit
    
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
    parser.add_argument("--dataset", type=str, default=None,
                       help="Custom dataset file path (overrides --easy)")
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
