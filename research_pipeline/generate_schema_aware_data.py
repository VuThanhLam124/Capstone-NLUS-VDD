"""
Generate Schema-aware Training Data
Creates training samples with full TPC-DS schema including:
- All columns with types
- Foreign key relationships
- Sample values for categorical columns
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# ========== CONFIG ==========
REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "research_pipeline" / "cache" / "ecommerce_dw.duckdb"
TRAIN_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "train_clean.csv"
OUTPUT_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "train_schema_aware.jsonl"

# ========== FOREIGN KEY RELATIONSHIPS ==========
FOREIGN_KEYS = {
    # store_sales FKs
    "store_sales.ss_sold_date_sk": "date_dim.d_date_sk",
    "store_sales.ss_item_sk": "item.i_item_sk",
    "store_sales.ss_customer_sk": "customer.c_customer_sk",
    "store_sales.ss_store_sk": "store.s_store_sk",
    "store_sales.ss_promo_sk": "promotion.p_promo_sk",
    
    # web_sales FKs
    "web_sales.ws_sold_date_sk": "date_dim.d_date_sk",
    "web_sales.ws_item_sk": "item.i_item_sk",
    "web_sales.ws_bill_customer_sk": "customer.c_customer_sk",
    "web_sales.ws_ship_customer_sk": "customer.c_customer_sk",
    "web_sales.ws_web_page_sk": "web_page.wp_web_page_sk",
    "web_sales.ws_web_site_sk": "web_site.web_site_sk",
    
    # catalog_sales FKs
    "catalog_sales.cs_sold_date_sk": "date_dim.d_date_sk",
    "catalog_sales.cs_item_sk": "item.i_item_sk",
    "catalog_sales.cs_bill_customer_sk": "customer.c_customer_sk",
    "catalog_sales.cs_ship_customer_sk": "customer.c_customer_sk",
    "catalog_sales.cs_catalog_page_sk": "catalog_page.cp_catalog_page_sk",
    
    # store_returns FKs
    "store_returns.sr_returned_date_sk": "date_dim.d_date_sk",
    "store_returns.sr_item_sk": "item.i_item_sk",
    "store_returns.sr_customer_sk": "customer.c_customer_sk",
    "store_returns.sr_store_sk": "store.s_store_sk",
    "store_returns.sr_reason_sk": "reason.r_reason_sk",
    
    # web_returns FKs
    "web_returns.wr_returned_date_sk": "date_dim.d_date_sk",
    "web_returns.wr_item_sk": "item.i_item_sk",
    "web_returns.wr_refunded_customer_sk": "customer.c_customer_sk",
    "web_returns.wr_reason_sk": "reason.r_reason_sk",
    
    # catalog_returns FKs
    "catalog_returns.cr_returned_date_sk": "date_dim.d_date_sk",
    "catalog_returns.cr_item_sk": "item.i_item_sk",
    "catalog_returns.cr_refunded_customer_sk": "customer.c_customer_sk",
    "catalog_returns.cr_reason_sk": "reason.r_reason_sk",
    
    # inventory FKs
    "inventory.inv_date_sk": "date_dim.d_date_sk",
    "inventory.inv_item_sk": "item.i_item_sk",
    "inventory.inv_warehouse_sk": "warehouse.w_warehouse_sk",
    
    # customer FKs
    "customer.c_current_addr_sk": "customer_address.ca_address_sk",
    "customer.c_current_cdemo_sk": "customer_demographics.cd_demo_sk",
    "customer.c_current_hdemo_sk": "household_demographics.hd_demo_sk",
}

# ========== SAMPLE VALUES ==========
SAMPLE_VALUES = {
    "ca_state": ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
    "ca_country": ["United States"],
    "cd_gender": ["M", "F"],
    "cd_marital_status": ["S", "M", "D", "W", "U"],
    "cd_education_status": ["Primary", "Secondary", "College", "2 yr Degree", "4 yr Degree", "Advanced Degree"],
    "i_category": ["Books", "Children", "Electronics", "Home", "Jewelry", "Men", "Music", "Shoes", "Sports", "Women"],
    "d_year": ["1998", "1999", "2000", "2001", "2002"],
    "d_moy": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    "d_qoy": ["1", "2", "3", "4"],
    "d_weekend": ["Y", "N"],
    "s_state": ["TN"],
    "w_state": ["TN"],
    "sm_type": ["EXPRESS", "OVERNIGHT", "REGULAR", "NEXT DAY", "LIBRARY"],
}

# ========== COLUMN DESCRIPTIONS ==========
COLUMN_DESCRIPTIONS = {
    # Primary keys - important to mark
    "c_customer_sk": "PRIMARY KEY",
    "i_item_sk": "PRIMARY KEY",
    "d_date_sk": "PRIMARY KEY",
    "s_store_sk": "PRIMARY KEY",
    "w_warehouse_sk": "PRIMARY KEY",
    "ca_address_sk": "PRIMARY KEY",
    "cd_demo_sk": "PRIMARY KEY",
    "hd_demo_sk": "PRIMARY KEY",
    "p_promo_sk": "PRIMARY KEY",
    "r_reason_sk": "PRIMARY KEY",
    "wp_web_page_sk": "PRIMARY KEY",
    "web_site_sk": "PRIMARY KEY",
    "cp_catalog_page_sk": "PRIMARY KEY",
    "sm_ship_mode_sk": "PRIMARY KEY",
    
    # Commonly confused columns
    "cd_demo_sk": "PRIMARY KEY - NOT cd_customer_sk",
    "ca_state": "2-char state code (CA, TX, NY...)",
    "d_weekend": "Y or N",
    "cd_dep_count": "number of dependents",
    "hd_dep_count": "household dependents",
    "hd_vehicle_count": "number of vehicles",
}


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract table names from SQL."""
    tables = set()
    for match in re.finditer(r'\b(?:FROM|JOIN)\s+([a-z_]+)', sql, re.I):
        tables.add(match.group(1).lower())
    return tables


def build_schema_for_tables(tables: List[str], schema_map: dict) -> str:
    """Build detailed schema text for selected tables."""
    lines = []
    
    for table in tables:
        if table not in schema_map:
            continue
        
        cols = schema_map[table]
        lines.append(f"TABLE {table} (")
        
        for col, typ in cols:
            parts = [f"  {col} {typ}"]
            
            # Add FK info
            fk_key = f"{table}.{col}"
            if fk_key in FOREIGN_KEYS:
                target = FOREIGN_KEYS[fk_key]
                parts.append(f"  -- FK -> {target}")
            
            # Add PK/description
            if col in COLUMN_DESCRIPTIONS:
                parts.append(f"  -- {COLUMN_DESCRIPTIONS[col]}")
            
            # Add sample values
            if col in SAMPLE_VALUES:
                values = ", ".join(SAMPLE_VALUES[col][:5])
                parts.append(f"  -- Values: {values}")
            
            lines.append(", ".join(parts) if len(parts) == 1 else parts[0] + " " + " ".join(parts[1:]))
        
        lines.append(")")
        lines.append("")
    
    return "\n".join(lines).strip()


def build_join_hints(tables: List[str]) -> str:
    """Build JOIN hints for selected tables."""
    hints = []
    
    for fk, pk in FOREIGN_KEYS.items():
        fk_table, fk_col = fk.split(".")
        pk_table, pk_col = pk.split(".")
        
        if fk_table in tables and pk_table in tables:
            hints.append(f"  {fk_table} JOIN {pk_table} ON {fk_col} = {pk_col}")
    
    if hints:
        return "JOIN HINTS:\n" + "\n".join(hints[:5])  # Limit to 5 hints
    return ""


def generate_schema_aware_sample(question: str, sql: str, schema_map: dict) -> dict:
    """Generate a single schema-aware training sample."""
    # Extract tables from SQL
    tables = list(extract_tables_from_sql(sql))
    
    # Build detailed schema
    schema_text = build_schema_for_tables(tables, schema_map)
    
    # Build join hints
    join_hints = build_join_hints(tables)
    
    # System prompt
    system = """You are an expert SQL writer for DuckDB (TPC-DS schema).
IMPORTANT: Use exact column names as shown in schema. Common mistakes to avoid:
- Use cd_demo_sk (not cd_customer_sk) for customer_demographics
- Use ca_state (not d_state) for state - state is in customer_address, not date_dim
- Use d_weekend = 'Y' or 'N' for weekend check
- Use i_current_price (not i_price) for item price
Output ONLY valid SQL ending with semicolon."""

    # User content
    user = f"SCHEMA:\n{schema_text}"
    if join_hints:
        user += f"\n\n{join_hints}"
    user += f"\n\nQUESTION:\n{question}\n\nSQL:"
    
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": sql}
        ]
    }


def main():
    print("="*50)
    print("Generating Schema-aware Training Data")
    print("="*50)
    
    if not HAS_DUCKDB:
        print("ERROR: DuckDB not available")
        return
    
    # Setup DB
    if not DB_PATH.exists():
        print("Setting up TPC-DS database...")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(DB_PATH))
        con.execute("INSTALL tpcds; LOAD tpcds;")
        con.execute("CALL dsdgen(sf=1);")
        con.close()
    
    con = duckdb.connect(str(DB_PATH), read_only=True)
    
    # Build schema map
    schema_map = {}
    for (table_name,) in con.execute("SHOW TABLES").fetchall():
        cols = [(r[0], r[1]) for r in con.execute(f"DESCRIBE {table_name}").fetchall()]
        schema_map[table_name] = cols
    print(f"Schema: {len(schema_map)} tables")
    
    con.close()
    
    # Load training data
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = train_df.dropna(subset=["Transcription", "SQL Ground Truth"])
    print(f"Training samples: {len(train_df)}")
    
    # Generate schema-aware samples
    samples = []
    for idx, row in train_df.iterrows():
        question = row["Transcription"]
        sql = row["SQL Ground Truth"]
        
        try:
            sample = generate_schema_aware_sample(question, sql, schema_map)
            samples.append(sample)
        except Exception as e:
            print(f"Error at {idx}: {e}")
    
    print(f"Generated: {len(samples)} samples")
    
    # Save as JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Saved to: {OUTPUT_PATH}")
    
    # Show example
    print("\n" + "="*50)
    print("Example sample:")
    print("="*50)
    if samples:
        example = samples[0]
        print(f"System: {example['messages'][0]['content'][:200]}...")
        print(f"\nUser: {example['messages'][1]['content'][:500]}...")
        print(f"\nAssistant: {example['messages'][2]['content'][:200]}...")


if __name__ == "__main__":
    main()
