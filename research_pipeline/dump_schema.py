import duckdb

DB_PATH = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/data/ecommerce_dw.duckdb"
OUTPUT_FILE = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/db_schema.md"

def dump_schema():
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = con.sql("SHOW TABLES").fetchall()
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# TPC-DS Database Schema (E-commerce)\n\n")
        
        for (table_name,) in tables:
            f.write(f"## Table: `{table_name}`\n")
            columns = con.sql(f"DESCRIBE {table_name}").fetchall()
            # columns: column_name, column_type, null, key, default, extra
            f.write("| Column Name | Type |\n|---|---|\n")
            for col in columns:
                f.write(f"| {col[0]} | {col[1]} |\n")
            f.write("\n")
            
    print(f"Schema dumped to {OUTPUT_FILE}")

if __name__ == "__main__":
    dump_schema()
