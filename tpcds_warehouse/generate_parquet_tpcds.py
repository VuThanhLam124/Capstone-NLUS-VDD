#!/usr/bin/env python3
"""
TPC-DS Snowflake Schema - Parquet Export Script
================================================
Script này tạo dữ liệu TPC-DS bằng DuckDB extension và export ra Parquet files
theo cấu trúc Snowflake Schema (fact tables và dimension tables tách biệt).

Date: 2025-12-26
"""

import duckdb
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

SCALE_FACTOR = 1  # SF=1 tạo ~1GB dữ liệu

# Output directory structure
BASE_OUTPUT_DIR = Path(__file__).parent / "tpcds_sf1"
FACT_DIR = BASE_OUTPUT_DIR / "fact"
DIM_DIR = BASE_OUTPUT_DIR / "dim"
METADATA_DIR = BASE_OUTPUT_DIR / "metadata"

# Compression: ZSTD (tốt cho storage) hoặc SNAPPY (tốt cho speed)
COMPRESSION = "ZSTD"

# ============================================================================
# SNOWFLAKE SCHEMA CLASSIFICATION
# ============================================================================

# Fact Tables: Chứa các transactions/events với foreign keys đến dimension tables
FACT_TABLES = {
    "store_sales": {
        "description": "Bán hàng tại cửa hàng - Fact table chính",
        "grain": "Mỗi dòng = 1 line item trong 1 transaction",
        "fk_dims": ["date_dim", "time_dim", "item", "customer", "store", "promotion"]
    },
    "store_returns": {
        "description": "Trả hàng tại cửa hàng",
        "grain": "Mỗi dòng = 1 line item bị trả lại",
        "fk_dims": ["date_dim", "time_dim", "item", "customer", "store", "reason"]
    },
    "catalog_sales": {
        "description": "Bán hàng qua catalog (mail order)",
        "grain": "Mỗi dòng = 1 line item trong đơn catalog",
        "fk_dims": ["date_dim", "time_dim", "item", "customer", "catalog_page", "ship_mode", "warehouse", "promotion"]
    },
    "catalog_returns": {
        "description": "Trả hàng từ đơn catalog",
        "grain": "Mỗi dòng = 1 line item trả lại từ catalog",
        "fk_dims": ["date_dim", "time_dim", "item", "customer", "catalog_page", "reason"]
    },
    "web_sales": {
        "description": "Bán hàng online qua website",
        "grain": "Mỗi dòng = 1 line item trong đơn web",
        "fk_dims": ["date_dim", "time_dim", "item", "customer", "web_page", "web_site", "ship_mode", "warehouse", "promotion"]
    },
    "web_returns": {
        "description": "Trả hàng từ đơn online",
        "grain": "Mỗi dòng = 1 line item trả lại từ web",
        "fk_dims": ["date_dim", "time_dim", "item", "customer", "web_page", "reason"]
    },
    "inventory": {
        "description": "Periodic snapshot của tồn kho",
        "grain": "Mỗi dòng = tồn kho của 1 item tại 1 warehouse vào 1 ngày",
        "fk_dims": ["date_dim", "item", "warehouse"]
    }
}

# Dimension Tables: Mô tả các entities, normalized theo Snowflake pattern
DIM_TABLES = {
    # Core dimensions
    "date_dim": {
        "description": "Dimension ngày - Role-playing dimension (sold_date, ship_date, return_date)",
        "normalized_from": None,
        "rows_approx": 73049
    },
    "time_dim": {
        "description": "Dimension thời gian trong ngày",
        "normalized_from": None,
        "rows_approx": 86400
    },
    "item": {
        "description": "Sản phẩm - Chứa thông tin product, category, brand",
        "normalized_from": None,
        "rows_approx": 18000
    },
    
    # Customer hierarchy (Snowflake normalized)
    "customer": {
        "description": "Khách hàng - Link đến address và demographics",
        "normalized_from": None,
        "rows_approx": 100000
    },
    "customer_address": {
        "description": "Địa chỉ khách hàng - Normalized từ customer",
        "normalized_from": "customer",
        "rows_approx": 50000
    },
    "customer_demographics": {
        "description": "Demographics khách hàng (gender, education, marital status)",
        "normalized_from": "customer",
        "rows_approx": 1920
    },
    
    # Household hierarchy
    "household_demographics": {
        "description": "Thông tin hộ gia đình",
        "normalized_from": None,
        "rows_approx": 7200
    },
    "income_band": {
        "description": "Phân khúc thu nhập - Normalized từ household",
        "normalized_from": "household_demographics",
        "rows_approx": 20
    },
    
    # Channel dimensions
    "store": {
        "description": "Cửa hàng bán lẻ",
        "normalized_from": None,
        "rows_approx": 12
    },
    "catalog_page": {
        "description": "Trang trong catalog phát hành",
        "normalized_from": None,
        "rows_approx": 11718
    },
    "web_page": {
        "description": "Trang web",
        "normalized_from": None,
        "rows_approx": 60
    },
    "web_site": {
        "description": "Website",
        "normalized_from": None,
        "rows_approx": 30
    },
    
    # Operational dimensions
    "warehouse": {
        "description": "Kho hàng",
        "normalized_from": None,
        "rows_approx": 5
    },
    "ship_mode": {
        "description": "Phương thức vận chuyển",
        "normalized_from": None,
        "rows_approx": 20
    },
    "call_center": {
        "description": "Trung tâm chăm sóc khách hàng",
        "normalized_from": None,
        "rows_approx": 6
    },
    
    # Other dimensions
    "promotion": {
        "description": "Chương trình khuyến mãi",
        "normalized_from": None,
        "rows_approx": 300
    },
    "reason": {
        "description": "Lý do trả hàng",
        "normalized_from": None,
        "rows_approx": 35
    }
}


def setup_directories() -> None:
    """Tạo cấu trúc thư mục output."""
    print(" Tạo cấu trúc thư mục...")
    FACT_DIR.mkdir(parents=True, exist_ok=True)
    DIM_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {FACT_DIR}")
    print(f"   ✓ {DIM_DIR}")
    print(f"   ✓ {METADATA_DIR}")


def generate_tpcds_data(con: duckdb.DuckDBPyConnection) -> None:
    """Generate TPC-DS data using DuckDB extension."""
    print(f"\n🔧 Cài đặt và load TPC-DS extension...")
    con.sql("INSTALL tpcds;")
    con.sql("LOAD tpcds;")
    print("   ✓ TPC-DS extension đã sẵn sàng")
    
    # Check if data already exists
    tables = [t[0] for t in con.sql("SHOW TABLES").fetchall()]
    if 'store_sales' in tables:
        count = con.sql("SELECT COUNT(*) FROM store_sales").fetchone()[0]
        if count > 0:
            print(f"   ℹ  Dữ liệu đã tồn tại ({count:,} rows trong store_sales)")
            return
    
    print(f"\n⏳ Generating TPC-DS data (Scale Factor = {SCALE_FACTOR})...")
    print("   Điều này có thể mất vài phút...")
    
    start_time = time.time()
    con.sql(f"CALL dsdgen(sf={SCALE_FACTOR});")
    elapsed = time.time() - start_time
    
    print(f"   ✓ Hoàn thành trong {elapsed:.2f} giây")


def get_table_schema(con: duckdb.DuckDBPyConnection, table_name: str) -> List[Dict]:
    """Lấy schema của bảng."""
    result = con.sql(f"DESCRIBE {table_name}").fetchall()
    return [
        {"column_name": row[0], "column_type": row[1], "null": row[2], "key": row[3]}
        for row in result
    ]


def export_table_to_parquet(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    output_dir: Path,
    table_info: Dict
) -> Dict:
    """Export một bảng ra Parquet file."""
    output_path = output_dir / f"{table_name}.parquet"
    
    # Get row count before export
    row_count = con.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    # Get schema
    schema = get_table_schema(con, table_name)
    
    # Export to Parquet
    con.sql(f"""
        COPY {table_name} TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION '{COMPRESSION}')
    """)
    
    # Get file size
    file_size = output_path.stat().st_size
    
    return {
        "table_name": table_name,
        "output_path": str(output_path),
        "row_count": row_count,
        "column_count": len(schema),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "compression": COMPRESSION,
        "schema": schema,
        **table_info
    }


def export_all_tables(con: duckdb.DuckDBPyConnection) -> Tuple[List[Dict], List[Dict]]:
    """Export tất cả fact và dimension tables."""
    fact_metadata = []
    dim_metadata = []
    
    # Export Fact Tables
    print(f"\n Exporting {len(FACT_TABLES)} Fact Tables...")
    for table_name, info in FACT_TABLES.items():
        print(f"   → {table_name}...", end=" ", flush=True)
        try:
            meta = export_table_to_parquet(con, table_name, FACT_DIR, info)
            fact_metadata.append(meta)
            print(f"✓ ({meta['row_count']:,} rows, {meta['file_size_mb']} MB)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Export Dimension Tables
    print(f"\n Exporting {len(DIM_TABLES)} Dimension Tables...")
    for table_name, info in DIM_TABLES.items():
        print(f"   → {table_name}...", end=" ", flush=True)
        try:
            meta = export_table_to_parquet(con, table_name, DIM_DIR, info)
            dim_metadata.append(meta)
            print(f"✓ ({meta['row_count']:,} rows, {meta['file_size_mb']} MB)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return fact_metadata, dim_metadata


def save_metadata(fact_metadata: List[Dict], dim_metadata: List[Dict]) -> None:
    """Lưu metadata về schema và tables."""
    metadata = {
        "scale_factor": SCALE_FACTOR,
        "compression": COMPRESSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fact_tables": fact_metadata,
        "dim_tables": dim_metadata,
        "totals": {
            "fact_tables_count": len(fact_metadata),
            "dim_tables_count": len(dim_metadata),
            "total_fact_rows": sum(t["row_count"] for t in fact_metadata),
            "total_dim_rows": sum(t["row_count"] for t in dim_metadata),
            "total_size_mb": sum(t["file_size_mb"] for t in fact_metadata + dim_metadata)
        }
    }
    
    output_path = METADATA_DIR / "schema_info.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n Metadata saved to {output_path}")


def print_summary(fact_metadata: List[Dict], dim_metadata: List[Dict]) -> None:
    """In tổng kết."""
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    total_fact_rows = sum(t["row_count"] for t in fact_metadata)
    total_dim_rows = sum(t["row_count"] for t in dim_metadata)
    total_size = sum(t["file_size_mb"] for t in fact_metadata + dim_metadata)
    
    print(f"\n Fact Tables: {len(fact_metadata)} tables, {total_fact_rows:,} rows")
    print(f"📐 Dimension Tables: {len(dim_metadata)} tables, {total_dim_rows:,} rows")
    print(f"💾 Total Size: {total_size:.2f} MB")
    print(f"🗜️  Compression: {COMPRESSION}")
    
    print(f"\n Output Directory: {BASE_OUTPUT_DIR}")
    print(f"   ├── fact/     ({len(fact_metadata)} files)")
    print(f"   ├── dim/      ({len(dim_metadata)} files)")
    print(f"   └── metadata/ (schema_info.json)")


def main():
    """Main execution."""
    print("="*60)
    print(" TPC-DS Snowflake Schema - Parquet Export")
    print(f"   Scale Factor: {SCALE_FACTOR}")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Connect to DuckDB (in-memory for generation)
    print("\n🔌 Connecting to DuckDB...")
    con = duckdb.connect(":memory:")
    
    try:
        # Generate TPC-DS data
        generate_tpcds_data(con)
        
        # Export tables to Parquet
        fact_metadata, dim_metadata = export_all_tables(con)
        
        # Save metadata
        save_metadata(fact_metadata, dim_metadata)
        
        # Print summary
        print_summary(fact_metadata, dim_metadata)
        
        print("\n Export hoàn tất thành công!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
