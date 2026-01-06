"""
Schema-aware Finetuning Script
Uses training data with full TPC-DS schema for better SQL generation.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
import torch
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# ========== CONFIG ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Schema-aware Finetuning")
    parser.add_argument("--data", type=str, default="research_pipeline/datasets/train_schema_aware.jsonl",
                        help="Path to JSONL training data")
    parser.add_argument("--output", type=str, default="/kaggle/working/finetuned_schema_aware",
                        help="Output directory")
    parser.add_argument("--adapter", type=str, default="Ellbendls/Qwen-3-4b-Text_to_SQL",
                        help="Base adapter to continue training from")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (reduced due to longer context)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (lower for schema-aware)")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    return parser.parse_args()


# Valid TPC-DS tables for validation
VALID_TABLES = {
    'store_sales', 'store_returns', 'web_sales', 'web_returns', 
    'catalog_sales', 'catalog_returns', 'inventory',
    'customer', 'customer_address', 'customer_demographics',
    'item', 'date_dim', 'time_dim', 'store', 'warehouse',
    'web_site', 'web_page', 'call_center', 'catalog_page',
    'promotion', 'reason', 'ship_mode', 'household_demographics', 'income_band'
}

def validate_sql_sample(sql: str) -> bool:
    """Validate SQL doesn't contain hallucinated tables."""
    import re
    # Extract table names from SQL
    tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-z_]+)', sql, re.I))
    # Check all tables are valid
    for table in tables:
        if table.lower() not in VALID_TABLES:
            return False
    # Check for repetitive patterns (hallucination indicator)
    if re.search(r'(\.[a-z]+){4,}', sql):
        return False
    return True

def load_jsonl_data(data_path: str, tokenizer) -> Dataset:
    """Load JSONL training data and format for training with validation."""
    samples = []
    skipped = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            messages = sample.get("messages", [])
            
            # Validate SQL in assistant response
            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
            if not validate_sql_sample(assistant_msg):
                skipped += 1
                continue
            
            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Skip if text is too long (truncation causes issues)
            if len(tokenizer.encode(text)) > 1800:  # Leave room for generation
                skipped += 1
                continue
                
            samples.append({"text": text})
    
    print(f"Loaded {len(samples)} valid samples, skipped {skipped} invalid/long samples")
    return Dataset.from_list(samples)


def main():
    args = parse_args()
    
    print("="*60)
    print("Schema-aware Finetuning")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Adapter: {args.adapter}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get base model from adapter
    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_model_id = peft_config.base_model_name_or_path
    print(f"\nAdapter: {args.adapter}")
    print(f"Base model: {base_model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    dataset = load_jsonl_data(args.data, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load base model
    print(f"\nLoading base model...")
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
    
    # Resize embeddings
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Load existing adapter and continue training
    print(f"Loading adapter: {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.print_trainable_parameters()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training config - optimized to prevent hallucination
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=16,  # Increased for smaller batch
        learning_rate=args.lr,
        weight_decay=0.05,  # Increased regularization to prevent overfitting
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.5,  # Slightly higher for stability
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,  # Explicit max length
        packing=False,  # Disable packing to avoid context confusion
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train()
    
    # Save
    print(f"\nSaving model to {args.output}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    print("\n" + "="*60)
    print("Schema-aware Finetuning Complete!")
    print("="*60)
    print(f"Model saved to: {args.output}")
    print("\nTo use:")
    print(f'  model = PeftModel.from_pretrained(base_model, "{args.output}")')


if __name__ == "__main__":
    main()
