import duckdb
import os
import time

# Dynamic Path Resolution: ./setup_dw.py -> ../data/ecommerce_dw.duckdb
# This ensures it works regardless of where the script is run from,
# as long as the project structure remains consistent.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "ecommerce_dw.duckdb")

SCALE_FACTOR = 1  # 1GB for testing. Will scale to 50 later.

def setup_tpcds():
    print(f"Ensuring data directory exists at: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Initializing DuckDB at {DB_PATH}...")
    
    # Connect to DuckDB (creates file if not exists)
    con = duckdb.connect(DB_PATH)
    
    try:
        print("Installing TPC-DS extension...")
        con.sql("INSTALL tpcds;")
        con.sql("LOAD tpcds;")
        print("TPC-DS extension loaded.")
        
        # Check if data already exists to avoid re-generating
        # We check one table, e.g., 'call_center'
        tables = [t[0] for t in con.sql("SHOW TABLES").fetchall()]
        
        if 'call_center' in tables:
            print("Data already appears to exist. Skipping generation.")
            count = con.sql("SELECT COUNT(*) FROM call_center").fetchone()[0]
            print(f"Call Center row count: {count}")
        else:
            print(f"Generating TPC-DS data (Scale Factor = {SCALE_FACTOR})... This may take a while.")
            start_time = time.time()
            
            # dsdgen generates all tables
            con.sql(f"CALL dsdgen(sf={SCALE_FACTOR});")
            
            end_time = time.time()
            print(f"Data generation completed in {end_time - start_time:.2f} seconds.")
            
        print("\nVerifying Schema...")
        # Print list of tables
        tables = con.sql("SHOW TABLES").fetchall()
        print(f"Found {len(tables)} tables: {', '.join([t[0] for t in tables])}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    setup_tpcds()
