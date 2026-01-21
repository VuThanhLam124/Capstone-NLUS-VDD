#!/usr/bin/env python3
"""
Update Ground Truth: Replace SELECT * with specific columns
Then verify all SQL queries against DuckDB

Steps:
1. Load test_easy.csv
2. Find SELECT * cases
3. Replace with appropriate columns based on table
4. Verify each SQL runs and returns results
5. Save updated CSV
"""

import csv
import re
import duckdb
from pathlib import Path

# TPC-DS database path
DB_PATH = "research_pipeline/cache/ecommerce_dw.duckdb"

# Column mappings for each table when using SELECT *
COLUMN_MAPPINGS = {
    'customer': 'c_customer_id, c_first_name, c_last_name, c_email_address',
    'customer_demographics': 'cd_demo_sk, cd_gender, cd_marital_status, cd_education_status, cd_dep_count',
    'customer_address': 'ca_address_sk, ca_city, ca_state, ca_country',
    'item': 'i_item_sk, i_item_id, i_item_desc, i_current_price, i_category, i_brand',
    'store': 's_store_sk, s_store_id, s_store_name, s_city, s_state, s_manager',
    'store_sales': 'ss_ticket_number, ss_item_sk, ss_customer_sk, ss_quantity, ss_sales_price, ss_net_paid',
    'web_sales': 'ws_order_number, ws_item_sk, ws_bill_customer_sk, ws_quantity, ws_sales_price, ws_net_paid',
    'catalog_sales': 'cs_order_number, cs_item_sk, cs_bill_customer_sk, cs_quantity, cs_sales_price, cs_net_paid',
    'date_dim': 'd_date_sk, d_date, d_year, d_month_seq, d_moy',
    'time_dim': 't_time_sk, t_time, t_hour, t_minute',
    'household_demographics': 'hd_demo_sk, hd_income_band_sk, hd_buy_potential, hd_dep_count, hd_vehicle_count',
    'promotion': 'p_promo_sk, p_promo_id, p_promo_name, p_channel_email, p_channel_tv',
    'warehouse': 'w_warehouse_sk, w_warehouse_id, w_warehouse_name, w_state',
    'web_site': 'web_site_sk, web_site_id, web_name, web_rec_start_date',
    'inventory': 'inv_item_sk, inv_warehouse_sk, inv_quantity_on_hand, inv_date_sk',
}


def extract_table_name(sql: str) -> str:
    """Extract primary table name from SQL"""
    # Pattern: FROM table_name or FROM table_name alias
    match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def replace_select_star(sql: str) -> str:
    """Replace SELECT * with specific columns"""
    if 'SELECT *' not in sql and 'SELECT  *' not in sql:
        return sql
    
    table = extract_table_name(sql)
    if table and table in COLUMN_MAPPINGS:
        columns = COLUMN_MAPPINGS[table]
        # Replace SELECT * with columns
        new_sql = re.sub(
            r'SELECT\s+\*',
            f'SELECT {columns}',
            sql,
            flags=re.IGNORECASE
        )
        return new_sql
    
    return sql


def verify_sql(conn, sql: str, question: str) -> tuple:
    """Verify SQL runs and returns results"""
    try:
        result = conn.execute(sql).fetchall()
        if len(result) == 0:
            return False, "No results returned"
        return True, f"OK ({len(result)} rows)"
    except Exception as e:
        return False, str(e)[:100]


def main():
    input_path = Path("research_pipeline/datasets/test_easy.csv")
    output_path = Path("research_pipeline/datasets/test_easy_updated.csv")
    
    print(f"Loading: {input_path}")
    
    # Connect to DuckDB
    conn = duckdb.connect(DB_PATH, read_only=True)
    
    # Read CSV
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"Total rows: {len(rows)}")
    
    # Process each row
    updated_count = 0
    syntax_error_count = 0
    no_result_count = 0
    verified = []
    
    for i, row in enumerate(rows):
        original_sql = row['SQL Ground Truth']
        question = row['Transcription']
        
        # Replace SELECT * if needed
        new_sql = replace_select_star(original_sql)
        was_updated = new_sql != original_sql
        
        if was_updated:
            updated_count += 1
            print(f"\n--- Updated ID {row['ID']} ---")
            print(f"Q: {question[:60]}...")
            print(f"Before: SELECT * ...")
            print(f"After:  {new_sql.split('FROM')[0].strip()[:60]}...")
        
        # Verify SQL syntax (not results)
        try:
            result = conn.execute(new_sql).fetchall()
            if len(result) == 0:
                no_result_count += 1
                # Still valid SQL, just no data matching
            status = "OK"
        except Exception as e:
            error_msg = str(e)
            # Check if it's a syntax error
            if 'Parser Error' in error_msg or 'Binder Error' in error_msg or 'Catalog Error' in error_msg:
                syntax_error_count += 1
                print(f"\n❌ SYNTAX ERROR ID {row['ID']}: {error_msg[:80]}")
                print(f"   SQL: {new_sql[:100]}")
                # Keep original if new one has syntax error
                if was_updated:
                    try:
                        conn.execute(original_sql).fetchall()
                        print(f"   Reverting to original (syntax OK)")
                        new_sql = original_sql
                        updated_count -= 1
                    except:
                        pass  # Both have errors, keep new one
            status = "ERROR"
        
        verified.append({
            'ID': row['ID'],
            'Transcription': question,
            'SQL Ground Truth': new_sql
        })
    
    # Write output
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Transcription', 'SQL Ground Truth'])
        writer.writeheader()
        writer.writerows(verified)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total rows: {len(rows)}")
    print(f"  Updated (SELECT * replaced): {updated_count}")
    print(f"  Syntax errors: {syntax_error_count}")
    print(f"  No results (valid SQL, no data): {no_result_count}")
    print(f"  Saved to: {output_path}")
    
    conn.close()


if __name__ == "__main__":
    main()
