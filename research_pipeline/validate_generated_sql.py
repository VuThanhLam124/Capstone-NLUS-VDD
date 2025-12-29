import duckdb
import csv
import pandas as pd
from pathlib import Path

# Config
DB_PATH = Path("research_pipeline/cache/ecommerce_dw.duckdb")
DATA_PATH = Path("research_pipeline/datasets/train_merged.csv")

def setup_db():
    if not DB_PATH.exists():
        print(f"Setting up TPC-DS database at {DB_PATH}...")
        con = duckdb.connect(str(DB_PATH))
        con.execute("INSTALL tpcds; LOAD tpcds;")
        con.execute("CALL dsdgen(sf=1);") # Scale factor 1
        con.close()
    return duckdb.connect(str(DB_PATH), read_only=True)

def validate_sql(data_path):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    con = setup_db()
    
    total = len(df)
    errors = 0
    empty_results = 0
    success = 0
    
    print(f"Validating {total} queries...")
    
    error_log = []
    empty_log = []

    for idx, row in df.iterrows():
        sql = row["SQL Ground Truth"]
        query_id = row.get("ID", f"row_{idx}")
        
        try:
            # Run query
            res = con.execute(sql).fetchall()
            
            if not res: # Empty result
                empty_results += 1
                empty_log.append((query_id, sql))
            elif all(all(v is None for v in r) for r in res): # All values are None (unlikely but possible)
                 empty_results += 1
                 empty_log.append((query_id, sql))
            else:
                success += 1
                
        except Exception as e:
            errors += 1
            error_log.append((query_id, str(e), sql))
            
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total}...")

    con.close()
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Total Queries: {total}")
    print(f"Success (Valid Result): {success} ({success/total*100:.1f}%)")
    print(f"Empty/Null Results: {empty_results} ({empty_results/total*100:.1f}%)")
    print(f"Execution Errors: {errors} ({errors/total*100:.1f}%)")
    print("="*50)
    
    if errors > 0:
        print("\nTop 5 Errors:")
        for eid, err, sql in error_log[:5]:
            print(f"- [{eid}] {err}\n  SQL: {sql[:100]}...")

    if empty_results > 0:
        print("\nTop 5 Empty Results:")
        for eid, sql in empty_log[:5]:
            print(f"- [{eid}] SQL: {sql[:100]}...")

if __name__ == "__main__":
    validate_sql(DATA_PATH)
