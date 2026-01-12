#!/usr/bin/env python3
"""
Finetune Qwen3-Coder-30B-A3B for Text-to-SQL
Optimized for RTX 5880 Ada (48GB) / A6000 (48GB)

Model: Qwen/Qwen3-Coder-30B-A3B-Instruct
- 30.5B total params, 3.3B active (MoE)
- 128 experts, 8 activated per token
- 256K context native

Usage:
    # Full pipeline (finetune + benchmark)
    python finetune_qwen_coder.py --epochs 3 --easy --max-test-samples 15

    # Skip training, only benchmark
    python finetune_qwen_coder.py --skip-train --adapter ./qwen_coder_finetuned --easy

    # With schema linking
    python finetune_qwen_coder.py --skip-train --adapter ./qwen_coder_finetuned --easy --schema-linking --few-shot 3
    
    # Fast benchmark with vLLM (10-20x faster)
    python finetune_qwen_coder.py --skip-train --use-vllm --easy --schema-linking --few-shot 3
"""

import os
import sys
import json
import re
import time
import gc
import argparse
from pathlib import Path
from datetime import datetime

import torch
import duckdb
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, 
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

from business_rules import load_business_rules
from prompt_assets import (
    load_few_shot_examples,
    load_full_schema,
    load_valid_tables,
)

# Monkey-patch DynamicCache for DeepSeek-Coder-V2 compatibility
# DeepSeek uses old cache API, but new transformers changed method names
try:
    from transformers.cache_utils import DynamicCache
    _patched = False
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        _patched = True
    if not hasattr(DynamicCache, 'get_max_length'):
        DynamicCache.get_max_length = lambda self: None  # No max length for dynamic cache
        _patched = True
    if not hasattr(DynamicCache, 'get_usable_length'):
        DynamicCache.get_usable_length = lambda self, new_seq_len, layer_idx=0: self.get_seq_length()
        _patched = True
    if _patched:
        print("Patched DynamicCache for DeepSeek compatibility")
except ImportError:
    pass

# Import schema linking
try:
    from schema_linking import SchemaLinker
    HAS_SCHEMA_LINKING = True
except ImportError:
    HAS_SCHEMA_LINKING = False
    print("WARNING: schema_linking.py not found")

# ========== CONFIG ==========
MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_OUTPUT_DIR = "./qwen_coder_finetuned"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Qwen3-Coder-30B-A3B for Text-to-SQL")
    
    # Data paths
    parser.add_argument("--train-data", type=str, 
                        default="research_pipeline/datasets/train_schema_aware.jsonl",
                        help="Training data (JSONL with messages format)")
    parser.add_argument("--test-data", type=str, 
                        default="research_pipeline/datasets/test.csv",
                        help="Test data (CSV)")
    parser.add_argument("--db", type=str, 
                        default="research_pipeline/cache/ecommerce_dw.duckdb",
                        help="Database path")
    
    # Model
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Base model")
    parser.add_argument("--adapter", type=str, default=None, 
                        help="Existing adapter to load (skip training)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for finetuned model")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (4 for 48GB GPU)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    
    # Benchmark params
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only benchmark")
    parser.add_argument("--easy", action="store_true", help="Use easy test set (test_easy.csv)")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--few-shot", type=int, default=0, help="Number of few-shot examples (0-3)")
    parser.add_argument("--schema-linking", action="store_true", help="Use dynamic schema linking")
    
    # Quantization
    parser.add_argument("--8bit", dest="use_8bit", action="store_true", 
                        help="Use 8-bit quantization (slower but more accurate)")
    
    # vLLM for fast inference
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for fast inference (10-20x faster, benchmark only)")
    
    return parser.parse_args()


# ========== TPC-DS SCHEMA ==========
FULL_SCHEMA = load_full_schema()

# Few-shot examples với channel disambiguation
FEW_SHOT_EXAMPLES = load_few_shot_examples("finetune_qwen_coder")


# ========== VALID TABLES ==========
VALID_TABLES = load_valid_tables()


def get_gpu_memory():
    """Get GPU memory info"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        return f"Total: {total:.1f}GB | Allocated: {allocated:.1f}GB | Reserved: {reserved:.1f}GB"
    return "No CUDA"


# ========== DATA LOADING ==========
def validate_sql_sample(sql: str) -> bool:
    """Validate SQL uses only valid TPC-DS tables"""
    tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-z_]+)', sql, re.I))
    for table in tables:
        if table.lower() not in VALID_TABLES:
            return False
    return True


def load_train_data(data_path: str, tokenizer, max_seq_length: int = 2048) -> Dataset:
    """Load training data from JSONL"""
    samples = []
    skipped = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("messages", [])
            
            # Validate SQL
            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
            if not validate_sql_sample(assistant_msg):
                skipped += 1
                continue
            
            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Skip if too long
            if len(tokenizer.encode(text)) > max_seq_length - 100:
                skipped += 1
                continue
            
            samples.append({"text": text})
    
    print(f"Loaded {len(samples)} valid samples, skipped {skipped}")
    return Dataset.from_list(samples)


# ========== MODEL LOADING ==========
def load_model_and_tokenizer(args):
    """Load Qwen3-Coder with quantization"""
    print(f"\n{'='*60}")
    print(f"Loading Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {get_gpu_memory()}")
    print(f"{'='*60}")
    
    # Quantization config
    if args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit NF4 quantization")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if flash attention is available
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print("Using FlashAttention2")
    except ImportError:
        attn_impl = "eager"
        print("FlashAttention2 not available, using eager attention")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    
    print(f"Model loaded! Memory: {get_gpu_memory()}")
    
    # Load existing adapter if specified
    if args.adapter and Path(args.adapter).exists():
        print(f"Loading adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=not args.skip_train)
    
    return model, tokenizer


def load_vllm_model(args):
    """Load model with vLLM for fast inference"""
    from vllm import LLM, SamplingParams
    
    print(f"\n{'='*60}")
    print(f"Loading Model with vLLM: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {get_gpu_memory()}")
    print(f"{'='*60}")
    
    # vLLM config
    vllm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
        "dtype": "bfloat16",
    }
    
    # Add LoRA adapter if specified
    if args.adapter and Path(args.adapter).exists():
        print(f"Loading with LoRA adapter: {args.adapter}")
        vllm_kwargs["enable_lora"] = True
        vllm_kwargs["max_lora_rank"] = 64
    
    llm = LLM(**vllm_kwargs)
    
    # Load tokenizer separately for chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"vLLM loaded! Memory: {get_gpu_memory()}")
    
    return llm, tokenizer


# ========== TRAINING ==========
def train_model(args, model, tokenizer):
    """Finetune with LoRA"""
    print(f"\n{'='*60}")
    print("PHASE 1: FINETUNING")
    print(f"{'='*60}")
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config for MoE model
    # Target all linear layers in attention and MLP
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",     # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    dataset = load_train_data(args.train_data, tokenizer, args.max_seq_length)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
    
    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        logging_dir=str(output_dir / "logs"),
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        dataset_text_field="text",
        report_to="none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to: {args.output}")
    
    return model


# ========== BENCHMARKING ==========
def setup_db(db_path: str):
    """Setup DuckDB connection"""
    conn = duckdb.connect(db_path, read_only=True)
    return conn


def postprocess_sql(sql: str) -> str:
    """Clean and fix generated SQL"""
    # Extract SQL from response
    sql = sql.strip()
    
    # Remove markdown code blocks
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0]
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0]
    
    # Remove comments and extra text
    lines = []
    for line in sql.split('\n'):
        line = line.strip()
        if line and not line.startswith('--') and not line.startswith('#'):
            lines.append(line)
    sql = '\n'.join(lines)
    
    # Fix common errors
    # Double prefix fix
    sql = re.sub(r'\bsr_sr_', 'sr_', sql)
    sql = re.sub(r'\bss_ss_', 'ss_', sql)
    sql = re.sub(r'\bws_ws_', 'ws_', sql)
    sql = re.sub(r'\bcs_cs_', 'cs_', sql)
    sql = re.sub(r'\bcr_cr_', 'cr_', sql)
    sql = re.sub(r'\bwr_wr_', 'wr_', sql)
    sql = re.sub(r'\br_r_', 'r_', sql)
    
    # Column name fixes
    sql = re.sub(r'\bd_quarter\b', 'd_qoy', sql)
    sql = re.sub(r'\bd_weekday\b', 'd_day_name', sql)
    sql = re.sub(r'\bd_wday\b', 'd_day_name', sql)
    
    # TOP N -> LIMIT N
    top_match = re.search(r'\bTOP\s+(\d+)\b', sql, re.I)
    if top_match:
        limit_val = top_match.group(1)
        sql = re.sub(r'\bTOP\s+\d+\b', '', sql, flags=re.I)
        if 'LIMIT' not in sql.upper():
            sql = sql.rstrip(';') + f' LIMIT {limit_val};'
    
    # Fix YEAR() function for DuckDB
    sql = re.sub(r'\bYEAR\s*\(\s*([^)]+)\s*\)', r'EXTRACT(YEAR FROM \1)', sql, flags=re.I)
    
    # Ensure ends with semicolon
    sql = sql.strip()
    if sql and not sql.endswith(';'):
        sql += ';'
    
    return sql


def build_prompt(question: str, schema_linker=None, few_shot: int = 0) -> str:
    """Build prompt with schema and optional few-shot examples"""
    
    # Use dynamic schema if available
    if schema_linker:
        schema = schema_linker.build_dynamic_schema(question, max_tables=5)
    else:
        schema = FULL_SCHEMA
    
    rules = load_business_rules()
    system_rules = "\n".join(
        [
            "Bạn là chuyên gia SQL cho TPC-DS database. Sinh câu SQL chính xác.",
            "",
            rules,
        ]
    )

    prompt_parts = [
        system_rules,
        "",
        "DATABASE SCHEMA:",
        schema,
    ]
    
    # Add few-shot examples
    if few_shot > 0:
        prompt_parts.append("\nEXAMPLES:")
        for i, ex in enumerate(FEW_SHOT_EXAMPLES[:few_shot]):
            prompt_parts.append(f"\nQ: {ex['question']}")
            prompt_parts.append(f"SQL:\n{ex['sql']}")
    
    prompt_parts.extend([
        "",
        f"QUESTION: {question}",
        "",
        "SQL:"
    ])
    
    return "\n".join(prompt_parts)


def generate_sql(model, tokenizer, question: str, schema_linker=None, few_shot: int = 0) -> str:
    """Generate SQL for question"""
    prompt = build_prompt(question, schema_linker, few_shot)
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # DeepSeek-Coder-V2 has KV cache issues with eager attention
    # Disable use_cache for DeepSeek models when FlashAttention is not available
    model_name = getattr(model.config, '_name_or_path', '') or ''
    is_deepseek = 'deepseek' in model_name.lower()
    use_cache = not is_deepseek  # Disable for DeepSeek due to attention size mismatch
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return postprocess_sql(response)


def generate_sql_vllm(llm, tokenizer, question: str, schema_linker=None, few_shot: int = 0, 
                       adapter_path: str = None) -> str:
    """Generate SQL using vLLM for fast inference"""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    
    prompt = build_prompt(question, schema_linker, few_shot)
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.05,
    )
    
    # Use LoRA adapter if specified
    lora_request = None
    if adapter_path and Path(adapter_path).exists():
        lora_request = LoRARequest("finetuned", 1, adapter_path)
    
    outputs = llm.generate([text], sampling_params, lora_request=lora_request)
    response = outputs[0].outputs[0].text
    
    return postprocess_sql(response)


def benchmark_model(args, model, tokenizer, use_vllm: bool = False):
    """Run benchmark on test set"""
    print(f"\n{'='*60}")
    print(f"PHASE 2: BENCHMARKING {'(vLLM)' if use_vllm else '(HuggingFace)'}")
    print(f"{'='*60}")
    
    # Load test data
    if args.easy:
        test_path = "research_pipeline/datasets/test_easy.csv"
    else:
        test_path = args.test_data
    
    test_df = pd.read_csv(test_path)
    print(f"Test set: {test_path} ({len(test_df)} samples)")
    
    if args.max_test_samples:
        test_df = test_df.head(args.max_test_samples)
        print(f"Using first {len(test_df)} samples")
    
    # Setup schema linker
    schema_linker = None
    if args.schema_linking and HAS_SCHEMA_LINKING:
        print("Loading schema linker...")
        schema_linker = SchemaLinker()
    
    # Setup DB
    conn = setup_db(args.db)
    
    # Detect column names
    q_col = "Transcription" if "Transcription" in test_df.columns else "question"
    sql_col = "SQL Ground Truth" if "SQL Ground Truth" in test_df.columns else "sql"
    
    # Run benchmark
    results = []
    valid_count = 0
    exec_match_count = 0
    
    for idx, row in test_df.iterrows():
        question = row[q_col]
        ground_truth = row[sql_col]
        
        print(f"\n[{idx+1}/{len(test_df)}] {question[:60]}...")
        
        start_time = time.time()
        if use_vllm:
            generated_sql = generate_sql_vllm(model, tokenizer, question, schema_linker, 
                                               args.few_shot, args.adapter)
        else:
            generated_sql = generate_sql(model, tokenizer, question, schema_linker, args.few_shot)
        gen_time = (time.time() - start_time) * 1000
        
        # Validate SQL
        is_valid = False
        exec_match = False
        error = None
        
        try:
            gen_result = conn.execute(generated_sql).fetchall()
            is_valid = True
            valid_count += 1
            
            # Check execution match
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
            print(f"  ❌ SQL Error: {error[:100]}")
        
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
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Valid SQL: {valid_count}/{len(test_df)} ({100*valid_count/len(test_df):.1f}%)")
    print(f"Exec Match: {exec_match_count}/{len(test_df)} ({100*exec_match_count/len(test_df):.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"research_pipeline/results/qwen_coder_benchmark_{timestamp}.json"
    Path("research_pipeline/results").mkdir(exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "valid_sql": valid_count,
            "exec_match": exec_match_count,
            "total": len(test_df),
            "model": args.model,
            "adapter": args.adapter,
            "few_shot": args.few_shot,
            "schema_linking": args.schema_linking,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return exec_match_count / len(test_df)


# ========== MAIN ==========
def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("Qwen3-Coder-30B-A3B Text-to-SQL Finetuning")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Backend: {'vLLM (fast)' if args.use_vllm else 'HuggingFace'}")
    
    # vLLM path: benchmark only
    if args.use_vllm:
        if not args.skip_train:
            print("WARNING: --use-vllm only supports benchmarking, adding --skip-train")
            args.skip_train = True
        
        # Load with vLLM
        model, tokenizer = load_vllm_model(args)
        
        # Benchmark
        accuracy = benchmark_model(args, model, tokenizer, use_vllm=True)
    else:
        # HuggingFace path: finetune + benchmark
        print(f"Training data: {args.train_data}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size} x {args.grad_accum} grad accum")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(args)
        
        # Phase 1: Training
        if not args.skip_train and not args.adapter:
            model = train_model(args, model, tokenizer)
            args.adapter = args.output
        elif args.skip_train:
            print("\nSkipping training (--skip-train)")
        
        # Phase 2: Benchmark
        accuracy = benchmark_model(args, model, tokenizer, use_vllm=False)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Final Accuracy: {accuracy*100:.1f}%")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
