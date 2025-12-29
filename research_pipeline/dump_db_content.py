import duckdb
import json
from pathlib import Path

# Config
DB_PATH = Path("research_pipeline/cache/ecommerce_dw.duckdb")
OUTPUT_PATH = Path("research_pipeline/datasets/db_content_samples.json")
MAX_SAMPLES = 5

def main():
    if not DB_PATH.exists():
        print("Database not found!")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    
    db_content = {}
    
    print(f"Sampling content from {len(tables)} tables...")
    
    for table in tables:
        # Get string columns only
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        string_cols = [c[0] for c in cols if "VARCHAR" in c[1] or "CHAR" in c[1]]
        
        table_samples = {}
        for col in string_cols:
            # Get distinct non-null values
            query = f"""
                SELECT DISTINCT {col} 
                FROM {table} 
                WHERE {col} IS NOT NULL 
                ORDER BY RANDOM() 
                LIMIT {MAX_SAMPLES}
            """
            try:
                samples = [r[0] for r in con.execute(query).fetchall()]
                if samples:
                    table_samples[col] = samples
            except Exception as e:
                print(f"Skipping {table}.{col}: {e}")
        
        if table_samples:
            db_content[table] = table_samples
            
    con.close()
    
    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(db_content, f, indent=2, ensure_ascii=False)
        
    print(f"Saved content samples to {OUTPUT_PATH}")
    # Preview
    print("Preview (item table):", json.dumps(db_content.get("item", {}), indent=2)[:500])

if __name__ == "__main__":
    main()
