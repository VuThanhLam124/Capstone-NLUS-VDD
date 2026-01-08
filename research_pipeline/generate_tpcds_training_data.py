"""
Generate TPC-DS Training Data with Schema Linking
Creates high-quality training data using:
1. Existing Q-SQL pairs from train.csv
2. Dynamic schema linking (context engineering)
3. Bidirectional schema verification
"""
import json
import pandas as pd
from pathlib import Path
from schema_linking import SchemaLinker, TPCDS_TABLES

def generate_tpcds_training_data(
    input_csv: str = "research_pipeline/datasets/train_clean.csv",
    output_jsonl: str = "research_pipeline/datasets/train_tpcds_schema_linked.jsonl",
    use_schema_linking: bool = True
):
    """
    Generate training data with proper TPC-DS schema context
    """
    # Load existing data
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["Transcription", "SQL Ground Truth"])
    
    # Initialize schema linker
    linker = SchemaLinker() if use_schema_linking else None
    
    samples = []
    skipped = 0
    
    for idx, row in df.iterrows():
        question = row["Transcription"]
        sql = row["SQL Ground Truth"]
        
        # Build schema context
        if use_schema_linking and linker:
            # Dynamic schema via linking
            schema_text = linker.build_dynamic_schema(question, max_tables=5)
            schema_method = "schema_linked"
        else:
            # Full schema (compact format)
            schema_text = build_compact_schema()
            schema_method = "full_schema"
        
        # Build chat messages
        system_message = """You are an expert SQL writer for DuckDB (TPC-DS schema).
CRITICAL RULES:
1. Use ONLY exact table and column names from SCHEMA
2. Use LIMIT N (NOT "TOP N" - that's SQL Server syntax)
3. Use CURRENT_DATE (NOT getdate())
4. JOIN dimension tables properly with _sk foreign keys
Output ONLY valid SQL ending with semicolon. No explanations."""
        
        user_message = f"SCHEMA:\n{schema_text}\n\nQUESTION:\n{question}\n\nSQL:"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": sql}
        ]
        
        samples.append({
            "messages": messages,
            "metadata": {
                "question": question,
                "schema_method": schema_method,
                "source": "train_clean"
            }
        })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples...")
    
    # Save to JSONL
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print(f"Training Data Generation Complete")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {output_path}")
    print(f"Schema method: {'Schema Linking' if use_schema_linking else 'Full Schema'}")


def build_compact_schema() -> str:
    """Build compact schema from TPCDS_TABLES"""
    lines = []
    for table_name, table_info in TPCDS_TABLES.items():
        cols = ", ".join(table_info["columns"])
        lines.append(f"TABLE {table_name} ({cols})")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate TPC-DS training data")
    parser.add_argument("--input", type=str, default="research_pipeline/datasets/train_clean.csv",
                        help="Input CSV file")
    parser.add_argument("--output", type=str, default="research_pipeline/datasets/train_tpcds_schema_linked.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--no-linking", action="store_true",
                        help="Disable schema linking (use full schema)")
    
    args = parser.parse_args()
    
    generate_tpcds_training_data(
        input_csv=args.input,
        output_jsonl=args.output,
        use_schema_linking=not args.no_linking
    )
