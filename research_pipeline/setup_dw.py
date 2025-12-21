import duckdb
import os
import time

DB_PATH = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/data/ecommerce_dw.duckdb"
SCALE_FACTOR = 1  # 1GB for testing. Will scale to 50 later.

def setup_tpcds():
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
        tables = con.sql("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if 'call_center' in table_names:
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
