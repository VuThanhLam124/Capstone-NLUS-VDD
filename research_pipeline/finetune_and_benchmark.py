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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return extract_sql(tokenizer.decode(gen_ids, skip_special_tokens=True))

def build_simple_prompt(question: str, tokenizer) -> str:
    system = "You are an expert SQL writer for DuckDB (TPC-DS schema). Output ONLY valid SQL ending with semicolon."
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
