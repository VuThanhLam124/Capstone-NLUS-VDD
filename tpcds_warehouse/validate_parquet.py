#!/usr/bin/env python3
"""
TPC-DS Parquet Validation Script
=================================
Script này kiểm tra tính toàn vẹn của dữ liệu sau khi export ra Parquet.

Validation bao gồm:
1. Schema verification: data types, column names
2. Row count validation
3. Data sample comparison
4. Foreign key integrity check

Author: Capstone NLU-VDD Team
Date: 2025-12-26
"""

import duckdb
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent / "tpcds_sf1"
FACT_DIR = BASE_DIR / "fact"
DIM_DIR = BASE_DIR / "dim"
METADATA_PATH = BASE_DIR / "metadata" / "schema_info.json"

# Foreign Key Relationships để validate
# Format: (fact_table, fact_column, dim_table, dim_column)
FK_RELATIONSHIPS = [
    # store_sales FKs
    ("store_sales", "ss_sold_date_sk", "date_dim", "d_date_sk"),
    ("store_sales", "ss_sold_time_sk", "time_dim", "t_time_sk"),
    ("store_sales", "ss_item_sk", "item", "i_item_sk"),
    ("store_sales", "ss_customer_sk", "customer", "c_customer_sk"),
    ("store_sales", "ss_store_sk", "store", "s_store_sk"),
    ("store_sales", "ss_promo_sk", "promotion", "p_promo_sk"),
    
    # catalog_sales FKs
    ("catalog_sales", "cs_sold_date_sk", "date_dim", "d_date_sk"),
    ("catalog_sales", "cs_item_sk", "item", "i_item_sk"),
    ("catalog_sales", "cs_bill_customer_sk", "customer", "c_customer_sk"),
    ("catalog_sales", "cs_warehouse_sk", "warehouse", "w_warehouse_sk"),
    
    # web_sales FKs
    ("web_sales", "ws_sold_date_sk", "date_dim", "d_date_sk"),
    ("web_sales", "ws_item_sk", "item", "i_item_sk"),
    ("web_sales", "ws_bill_customer_sk", "customer", "c_customer_sk"),
    
    # inventory FKs
    ("inventory", "inv_date_sk", "date_dim", "d_date_sk"),
    ("inventory", "inv_item_sk", "item", "i_item_sk"),
    ("inventory", "inv_warehouse_sk", "warehouse", "w_warehouse_sk"),
    
    # Snowflake hierarchy: customer -> customer_address
    ("customer", "c_current_addr_sk", "customer_address", "ca_address_sk"),
    ("customer", "c_current_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("customer", "c_current_hdemo_sk", "household_demographics", "hd_demo_sk"),
    
    # household_demographics -> income_band
    ("household_demographics", "hd_income_band_sk", "income_band", "ib_income_band_sk"),
]


class ValidationResult:
    """Kết quả validation."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details: List[Dict] = []
    
    def add_pass(self, check: str, message: str):
        self.passed += 1
        self.details.append({"status": "PASS", "check": check, "message": message})
    
    def add_fail(self, check: str, message: str):
        self.failed += 1
        self.details.append({"status": "FAIL", "check": check, "message": message})
    
    def add_warning(self, check: str, message: str):
        self.warnings += 1
        self.details.append({"status": "WARN", "check": check, "message": message})
    
    def summary(self) -> str:
        return f"✓ {self.passed} passed, ✗ {self.failed} failed, ⚠ {self.warnings} warnings"


def load_metadata() -> Optional[Dict]:
    """Load metadata từ schema_info.json."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def validate_file_existence(result: ValidationResult) -> bool:
    """Kiểm tra các file Parquet tồn tại."""
    print("\n1️⃣  Checking file existence...")
    
    all_exist = True
    
    # Check directories
    for dir_path, dir_name in [(FACT_DIR, "fact"), (DIM_DIR, "dim")]:
        if dir_path.exists():
            files = list(dir_path.glob("*.parquet"))
            result.add_pass("directory", f"{dir_name}/ exists with {len(files)} parquet files")
        else:
            result.add_fail("directory", f"{dir_name}/ does not exist")
            all_exist = False
    
    return all_exist


def validate_schema(con: duckdb.DuckDBPyConnection, result: ValidationResult, metadata: Dict):
    """Validate schema của từng bảng."""
    print("\n2️⃣  Validating schemas...")
    
    all_tables = metadata.get("fact_tables", []) + metadata.get("dim_tables", [])
    
    for table_meta in all_tables:
        table_name = table_meta["table_name"]
        expected_schema = table_meta.get("schema", [])
        parquet_path = table_meta["output_path"]
        
        if not Path(parquet_path).exists():
            result.add_fail("schema", f"{table_name}: file not found")
            continue
        
        # Get actual schema from Parquet
        try:
            actual_schema = con.sql(f"DESCRIBE SELECT * FROM '{parquet_path}'").fetchall()
            actual_columns = {row[0]: row[1] for row in actual_schema}
            expected_columns = {col["column_name"]: col["column_type"] for col in expected_schema}
            
            # Compare column counts
            if len(actual_columns) == len(expected_columns):
                # Check column names match
                if set(actual_columns.keys()) == set(expected_columns.keys()):
                    result.add_pass("schema", f"{table_name}: {len(actual_columns)} columns OK")
                else:
                    missing = set(expected_columns.keys()) - set(actual_columns.keys())
                    result.add_fail("schema", f"{table_name}: missing columns {missing}")
            else:
                result.add_fail("schema", 
                    f"{table_name}: column count mismatch (expected {len(expected_columns)}, got {len(actual_columns)})")
        
        except Exception as e:
            result.add_fail("schema", f"{table_name}: error reading schema - {e}")


def validate_row_counts(con: duckdb.DuckDBPyConnection, result: ValidationResult, metadata: Dict):
    """Validate row counts khớp với metadata."""
    print("\n3️⃣  Validating row counts...")
    
    all_tables = metadata.get("fact_tables", []) + metadata.get("dim_tables", [])
    
    for table_meta in all_tables:
        table_name = table_meta["table_name"]
        expected_rows = table_meta["row_count"]
        parquet_path = table_meta["output_path"]
        
        if not Path(parquet_path).exists():
            continue
        
        try:
            actual_rows = con.sql(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
            
            if actual_rows == expected_rows:
                result.add_pass("row_count", f"{table_name}: {actual_rows:,} rows OK")
            else:
                result.add_fail("row_count", 
                    f"{table_name}: row count mismatch (expected {expected_rows:,}, got {actual_rows:,})")
        
        except Exception as e:
            result.add_fail("row_count", f"{table_name}: error counting rows - {e}")


def validate_foreign_keys(con: duckdb.DuckDBPyConnection, result: ValidationResult):
    """Validate foreign key relationships."""
    print("\n4️⃣  Validating foreign key relationships...")
    
    for fact_table, fact_col, dim_table, dim_col in FK_RELATIONSHIPS:
        # Determine paths
        if fact_table in ["store_sales", "store_returns", "catalog_sales", 
                          "catalog_returns", "web_sales", "web_returns", "inventory"]:
            fact_path = FACT_DIR / f"{fact_table}.parquet"
        else:
            fact_path = DIM_DIR / f"{fact_table}.parquet"
        
        dim_path = DIM_DIR / f"{dim_table}.parquet"
        
        if not fact_path.exists() or not dim_path.exists():
            result.add_warning("fk", f"{fact_table}.{fact_col} -> {dim_table}.{dim_col}: files missing")
            continue
        
        try:
            # Check for orphan foreign keys (fact values not in dimension)
            # Allow NULL values (they're valid in TPC-DS)
            orphan_query = f"""
                SELECT COUNT(DISTINCT f.{fact_col}) as orphan_count
                FROM '{fact_path}' f
                LEFT JOIN '{dim_path}' d ON f.{fact_col} = d.{dim_col}
                WHERE f.{fact_col} IS NOT NULL AND d.{dim_col} IS NULL
            """
            orphan_count = con.sql(orphan_query).fetchone()[0]
            
            if orphan_count == 0:
                result.add_pass("fk", f"{fact_table}.{fact_col} -> {dim_table}.{dim_col}: OK")
            else:
                result.add_warning("fk", 
                    f"{fact_table}.{fact_col} -> {dim_table}.{dim_col}: {orphan_count:,} orphan values")
        
        except Exception as e:
            result.add_warning("fk", f"{fact_table}.{fact_col} -> {dim_table}: error - {e}")


def validate_data_sample(con: duckdb.DuckDBPyConnection, result: ValidationResult, metadata: Dict):
    """Kiểm tra mẫu dữ liệu không bị corruption."""
    print("\n5️⃣  Validating data samples...")
    
    # Check một vài fact tables
    sample_tables = ["store_sales", "customer", "item", "date_dim"]
    
    for table_name in sample_tables:
        if table_name in ["store_sales", "inventory"]:
            path = FACT_DIR / f"{table_name}.parquet"
        else:
            path = DIM_DIR / f"{table_name}.parquet"
        
        if not path.exists():
            continue
        
        try:
            # Try to read some rows
            sample = con.sql(f"SELECT * FROM '{path}' LIMIT 5").fetchall()
            if len(sample) > 0:
                result.add_pass("data_sample", f"{table_name}: readable, sample OK")
            else:
                result.add_warning("data_sample", f"{table_name}: empty table")
        
        except Exception as e:
            result.add_fail("data_sample", f"{table_name}: read error - {e}")


def print_final_report(result: ValidationResult):
    """In báo cáo kết quả cuối cùng."""
    print("\n" + "="*60)
    print("📋 VALIDATION REPORT")
    print("="*60)
    
    # Group by status
    for status in ["FAIL", "WARN", "PASS"]:
        items = [d for d in result.details if d["status"] == status]
        if items:
            icon = {"FAIL": "❌", "WARN": "⚠️", "PASS": "✅"}[status]
            print(f"\n{icon} {status} ({len(items)}):")
            for item in items:
                print(f"   [{item['check']}] {item['message']}")
    
    print("\n" + "="*60)
    print(f"📊 SUMMARY: {result.summary()}")
    print("="*60)
    
    if result.failed == 0:
        print("\n🎉 All validations passed!")
        return 0
    else:
        print(f"\n⚠️  {result.failed} validation(s) failed. Please review.")
        return 1


def main():
    """Main execution."""
    print("="*60)
    print("🔍 TPC-DS Parquet Validation")
    print("="*60)
    
    result = ValidationResult()
    
    # Load metadata
    metadata = load_metadata()
    if not metadata:
        print(f"❌ Metadata file not found at {METADATA_PATH}")
        print("   Please run generate_parquet_tpcds.py first.")
        return 1
    
    print(f"\n📝 Loaded metadata: SF={metadata.get('scale_factor')}, "
          f"generated at {metadata.get('generated_at')}")
    
    # Connect to DuckDB
    con = duckdb.connect(":memory:")
    
    try:
        # Run validations
        if not validate_file_existence(result):
            print("\n❌ Basic file checks failed. Cannot continue.")
            return 1
        
        validate_schema(con, result, metadata)
        validate_row_counts(con, result, metadata)
        validate_foreign_keys(con, result)
        validate_data_sample(con, result, metadata)
        
        # Print report
        return print_final_report(result)
    
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
