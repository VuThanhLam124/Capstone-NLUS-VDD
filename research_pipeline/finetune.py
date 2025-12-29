"""
Finetune Text-to-SQL Model with LoRA
Uses QLoRA (4-bit quantization + LoRA) for efficient training on consumer GPUs.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# Try importing DataCollatorForCompletionOnlyLM from different locations
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.data_utils import DataCollatorForCompletionOnlyLM
    except ImportError:
        # If not available, we'll use None and skip completion-only training
        DataCollatorForCompletionOnlyLM = None
        print("WARNING: DataCollatorForCompletionOnlyLM not available, using full sequence training")

# ========== CONFIG (can be overridden via command line) ==========
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Text-to-SQL Model")
    parser.add_argument("--data", type=str, default="research_pipeline/datasets/train_clean.csv",
                        help="Path to training data CSV (columns: Transcription, SQL Ground Truth)")
    parser.add_argument("--output", type=str, default="/kaggle/working/finetuned_model",
                        help="Output directory for finetuned model")
    parser.add_argument("--adapter", type=str, default="Ellbendls/Qwen-3-4b-Text_to_SQL",
                        help="Existing adapter to continue training from")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    return parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent

# System prompt
SYSTEM_PROMPT = "You translate user questions into SQL for DuckDB (TPC-DS). Return only SQL, no markdown."

# ========== TABLE DESCRIPTIONS ==========
TABLE_DESCRIPTIONS = {
    "store_sales": "-- Store sales: ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_store_sk, ss_quantity, ss_net_paid",
    "web_sales": "-- Web sales: ws_sold_date_sk, ws_item_sk, ws_bill_customer_sk, ws_quantity, ws_net_paid",
    "catalog_sales": "-- Catalog sales: cs_sold_date_sk, cs_item_sk, cs_bill_customer_sk, cs_quantity, cs_net_paid",
    "store_returns": "-- Store returns: sr_returned_date_sk, sr_item_sk, sr_customer_sk, sr_return_amt",
    "web_returns": "-- Web returns: wr_returned_date_sk, wr_item_sk, wr_refunded_customer_sk, wr_return_amt",
    "catalog_returns": "-- Catalog returns: cr_returned_date_sk, cr_item_sk, cr_refunded_customer_sk, cr_return_amt",
    "inventory": "-- Inventory: inv_date_sk, inv_item_sk, inv_warehouse_sk, inv_quantity_on_hand",
    "customer": "-- Customer: c_customer_sk, c_first_name, c_last_name, c_current_addr_sk, c_current_cdemo_sk",
    "customer_address": "-- Address: ca_address_sk, ca_city, ca_state, ca_zip, ca_country",
    "customer_demographics": "-- Demographics: cd_demo_sk, cd_gender (M/F), cd_marital_status, cd_education_status",
    "item": "-- Item: i_item_sk, i_item_id, i_item_desc, i_category, i_brand, i_current_price",
    "date_dim": "-- Date: d_date_sk, d_date, d_year, d_moy (month 1-12), d_qoy (quarter 1-4)",
    "store": "-- Store: s_store_sk, s_store_name, s_city, s_state",
    "warehouse": "-- Warehouse: w_warehouse_sk, w_warehouse_name, w_city, w_state",
    "promotion": "-- Promotion: p_promo_sk, p_promo_name, p_channel_email, p_channel_tv",
    "reason": "-- Return reason: r_reason_sk, r_reason_id, r_reason_desc",
    "ship_mode": "-- Shipping: sm_ship_mode_sk, sm_type (EXPRESS/OVERNIGHT/REGULAR)",
    "household_demographics": "-- Household: hd_demo_sk, hd_income_band_sk, hd_buy_potential, hd_vehicle_count",
}

def get_schema_for_sql(sql: str) -> str:
    """Extract relevant schema based on tables in SQL."""
    import re
    tables_in_sql = set()
    for match in re.finditer(r'\b(?:FROM|JOIN)\s+([a-z_]+)', sql, re.I):
        tables_in_sql.add(match.group(1).lower())
    
    schema_lines = []
    for table in tables_in_sql:
        if table in TABLE_DESCRIPTIONS:
            schema_lines.append(TABLE_DESCRIPTIONS[table])
            schema_lines.append(f"TABLE {table} (...)")  # Simplified
    
    return "\n".join(schema_lines) if schema_lines else "TABLE store_sales (...)\nTABLE date_dim (...)"

def format_sample(row, tokenizer):
    """Format a training sample in chat format."""
    question = row["Transcription"]
    sql = row["SQL Ground Truth"]
    
    # Get relevant schema
    schema = get_schema_for_sql(sql)
    
    # Build prompt
    user_content = f"SCHEMA:\n{schema}\n\nQUESTION:\n{question}\n\nSQL:"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sql}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

def load_data(data_path, tokenizer):
    """Load and format training data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["Transcription", "SQL Ground Truth"])
    print(f"Loaded {len(df)} samples")
    
    # Convert to dataset
    dataset = Dataset.from_pandas(df)
    
    # Format for training
    dataset = dataset.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    return dataset

def main():
    args = parse_args()
    
    print("="*50)
    print("Continue Finetuning from Existing Adapter")
    print("="*50)
    print(f"\nConfig:")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print("="*50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Finetuning requires GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get base model from adapter config
    from peft import PeftConfig, PeftModel
    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_model_id = peft_config.base_model_name_or_path
    print(f"\nAdapter: {args.adapter}")
    print(f"Base model: {base_model_id}")
    
    # Load tokenizer from adapter
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    dataset = load_data(args.data, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load base model with 4-bit quantization
    print(f"\nLoading base model: {base_model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Changed from bfloat16
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Resize embeddings if needed
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Load existing adapter
    print(f"Loading adapter: {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.print_trainable_parameters()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments - use SFTConfig for newer TRL versions
    try:
        # TRL >= 0.8.0
        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
        )
    except:
        # Fallback for older TRL
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
        )
    
    # Data collator for completion-only training
    collator = None
    if DataCollatorForCompletionOnlyLM is not None:
        response_template = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\nFinetuning complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTo use the finetuned model:")
    print(f'  model = PeftModel.from_pretrained(base_model, "{OUTPUT_DIR}")')

if __name__ == "__main__":
    main()
