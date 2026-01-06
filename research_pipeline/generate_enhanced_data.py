"""
Generate finetune data with per-table schema context for TPC-DS.
Each sample includes:
- schema name and purpose
- features (column name + type)
- keys (primary key, foreign keys with references)
- question + simple SQL
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import duckdb
except ImportError as exc:
    raise SystemExit("duckdb is required to run this script") from exc

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "research_pipeline" / "cache" / "ecommerce_dw.duckdb"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "research_pipeline" / "datasets" / "train_schema_context.jsonl"

# Purpose descriptions based on TPC-DS table roles
TABLE_PURPOSES = {
    "store_sales": "Store channel sales transactions.",
    "web_sales": "Web channel sales transactions.",
    "catalog_sales": "Catalog channel sales transactions.",
    "store_returns": "Store channel return transactions.",
    "web_returns": "Web channel return transactions.",
    "catalog_returns": "Catalog channel return transactions.",
    "inventory": "Inventory snapshot by item and warehouse.",
    "customer": "Customer master data.",
    "customer_address": "Customer address details.",
    "customer_demographics": "Customer demographic attributes.",
    "household_demographics": "Household demographic attributes.",
    "income_band": "Income band ranges.",
    "item": "Product catalog details.",
    "date_dim": "Calendar date dimension.",
    "time_dim": "Time-of-day dimension.",
    "store": "Store location details.",
    "warehouse": "Warehouse location details.",
    "web_site": "Web site details.",
    "web_page": "Web page details.",
    "call_center": "Call center details.",
    "catalog_page": "Catalog page details.",
    "promotion": "Promotion details.",
    "reason": "Return reason codes.",
    "ship_mode": "Shipping mode details.",
}

PRIMARY_KEYS = {
    "customer": ["c_customer_sk"],
    "customer_address": ["ca_address_sk"],
    "customer_demographics": ["cd_demo_sk"],
    "household_demographics": ["hd_demo_sk"],
    "income_band": ["ib_income_band_sk"],
    "item": ["i_item_sk"],
    "date_dim": ["d_date_sk"],
    "time_dim": ["t_time_sk"],
    "store": ["s_store_sk"],
    "warehouse": ["w_warehouse_sk"],
    "web_site": ["web_site_sk"],
    "web_page": ["wp_web_page_sk"],
    "call_center": ["cc_call_center_sk"],
    "catalog_page": ["cp_catalog_page_sk"],
    "promotion": ["p_promo_sk"],
    "reason": ["r_reason_sk"],
    "ship_mode": ["sm_ship_mode_sk"],
}

# Candidate foreign keys based on TPC-DS schema
FOREIGN_KEY_CANDIDATES = [
    # store_sales
    ("store_sales", "ss_sold_date_sk", "date_dim", "d_date_sk"),
    ("store_sales", "ss_sold_time_sk", "time_dim", "t_time_sk"),
    ("store_sales", "ss_item_sk", "item", "i_item_sk"),
    ("store_sales", "ss_customer_sk", "customer", "c_customer_sk"),
    ("store_sales", "ss_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("store_sales", "ss_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("store_sales", "ss_addr_sk", "customer_address", "ca_address_sk"),
    ("store_sales", "ss_store_sk", "store", "s_store_sk"),
    ("store_sales", "ss_promo_sk", "promotion", "p_promo_sk"),
    # store_returns
    ("store_returns", "sr_returned_date_sk", "date_dim", "d_date_sk"),
    ("store_returns", "sr_return_time_sk", "time_dim", "t_time_sk"),
    ("store_returns", "sr_item_sk", "item", "i_item_sk"),
    ("store_returns", "sr_customer_sk", "customer", "c_customer_sk"),
    ("store_returns", "sr_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("store_returns", "sr_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("store_returns", "sr_addr_sk", "customer_address", "ca_address_sk"),
    ("store_returns", "sr_store_sk", "store", "s_store_sk"),
    ("store_returns", "sr_reason_sk", "reason", "r_reason_sk"),
    # web_sales
    ("web_sales", "ws_sold_date_sk", "date_dim", "d_date_sk"),
    ("web_sales", "ws_sold_time_sk", "time_dim", "t_time_sk"),
    ("web_sales", "ws_ship_date_sk", "date_dim", "d_date_sk"),
    ("web_sales", "ws_item_sk", "item", "i_item_sk"),
    ("web_sales", "ws_bill_customer_sk", "customer", "c_customer_sk"),
    ("web_sales", "ws_bill_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("web_sales", "ws_bill_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("web_sales", "ws_bill_addr_sk", "customer_address", "ca_address_sk"),
    ("web_sales", "ws_ship_customer_sk", "customer", "c_customer_sk"),
    ("web_sales", "ws_ship_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("web_sales", "ws_ship_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("web_sales", "ws_ship_addr_sk", "customer_address", "ca_address_sk"),
    ("web_sales", "ws_web_page_sk", "web_page", "wp_web_page_sk"),
    ("web_sales", "ws_web_site_sk", "web_site", "web_site_sk"),
    ("web_sales", "ws_ship_mode_sk", "ship_mode", "sm_ship_mode_sk"),
    ("web_sales", "ws_warehouse_sk", "warehouse", "w_warehouse_sk"),
    ("web_sales", "ws_promo_sk", "promotion", "p_promo_sk"),
    # web_returns
    ("web_returns", "wr_returned_date_sk", "date_dim", "d_date_sk"),
    ("web_returns", "wr_returned_time_sk", "time_dim", "t_time_sk"),
    ("web_returns", "wr_item_sk", "item", "i_item_sk"),
    ("web_returns", "wr_refunded_customer_sk", "customer", "c_customer_sk"),
    ("web_returns", "wr_refunded_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("web_returns", "wr_refunded_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("web_returns", "wr_refunded_addr_sk", "customer_address", "ca_address_sk"),
    ("web_returns", "wr_returning_customer_sk", "customer", "c_customer_sk"),
    ("web_returns", "wr_returning_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("web_returns", "wr_returning_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("web_returns", "wr_returning_addr_sk", "customer_address", "ca_address_sk"),
    ("web_returns", "wr_web_page_sk", "web_page", "wp_web_page_sk"),
    ("web_returns", "wr_reason_sk", "reason", "r_reason_sk"),
    # catalog_sales
    ("catalog_sales", "cs_sold_date_sk", "date_dim", "d_date_sk"),
    ("catalog_sales", "cs_sold_time_sk", "time_dim", "t_time_sk"),
    ("catalog_sales", "cs_ship_date_sk", "date_dim", "d_date_sk"),
    ("catalog_sales", "cs_bill_customer_sk", "customer", "c_customer_sk"),
    ("catalog_sales", "cs_bill_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("catalog_sales", "cs_bill_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("catalog_sales", "cs_bill_addr_sk", "customer_address", "ca_address_sk"),
    ("catalog_sales", "cs_ship_customer_sk", "customer", "c_customer_sk"),
    ("catalog_sales", "cs_ship_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("catalog_sales", "cs_ship_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("catalog_sales", "cs_ship_addr_sk", "customer_address", "ca_address_sk"),
    ("catalog_sales", "cs_call_center_sk", "call_center", "cc_call_center_sk"),
    ("catalog_sales", "cs_catalog_page_sk", "catalog_page", "cp_catalog_page_sk"),
    ("catalog_sales", "cs_ship_mode_sk", "ship_mode", "sm_ship_mode_sk"),
    ("catalog_sales", "cs_warehouse_sk", "warehouse", "w_warehouse_sk"),
    ("catalog_sales", "cs_item_sk", "item", "i_item_sk"),
    ("catalog_sales", "cs_promo_sk", "promotion", "p_promo_sk"),
    # catalog_returns
    ("catalog_returns", "cr_returned_date_sk", "date_dim", "d_date_sk"),
    ("catalog_returns", "cr_returned_time_sk", "time_dim", "t_time_sk"),
    ("catalog_returns", "cr_item_sk", "item", "i_item_sk"),
    ("catalog_returns", "cr_refunded_customer_sk", "customer", "c_customer_sk"),
    ("catalog_returns", "cr_refunded_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("catalog_returns", "cr_refunded_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("catalog_returns", "cr_refunded_addr_sk", "customer_address", "ca_address_sk"),
    ("catalog_returns", "cr_returning_customer_sk", "customer", "c_customer_sk"),
    ("catalog_returns", "cr_returning_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("catalog_returns", "cr_returning_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("catalog_returns", "cr_returning_addr_sk", "customer_address", "ca_address_sk"),
    ("catalog_returns", "cr_call_center_sk", "call_center", "cc_call_center_sk"),
    ("catalog_returns", "cr_catalog_page_sk", "catalog_page", "cp_catalog_page_sk"),
    ("catalog_returns", "cr_ship_mode_sk", "ship_mode", "sm_ship_mode_sk"),
    ("catalog_returns", "cr_warehouse_sk", "warehouse", "w_warehouse_sk"),
    ("catalog_returns", "cr_reason_sk", "reason", "r_reason_sk"),
    # inventory
    ("inventory", "inv_date_sk", "date_dim", "d_date_sk"),
    ("inventory", "inv_item_sk", "item", "i_item_sk"),
    ("inventory", "inv_warehouse_sk", "warehouse", "w_warehouse_sk"),
    # customer
    ("customer", "c_current_addr_sk", "customer_address", "ca_address_sk"),
    ("customer", "c_current_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    ("customer", "c_current_hdemo_sk", "household_demographics", "hd_demo_sk"),
    ("customer", "c_first_shipto_date_sk", "date_dim", "d_date_sk"),
    ("customer", "c_first_sales_date_sk", "date_dim", "d_date_sk"),
    ("customer", "c_last_review_date_sk", "date_dim", "d_date_sk"),
    # household_demographics
    ("household_demographics", "hd_income_band_sk", "income_band", "ib_income_band_sk"),
    # store
    ("store", "s_closed_date_sk", "date_dim", "d_date_sk"),
    # call_center
    ("call_center", "cc_closed_date_sk", "date_dim", "d_date_sk"),
    ("call_center", "cc_open_date_sk", "date_dim", "d_date_sk"),
    # catalog_page
    ("catalog_page", "cp_start_date_sk", "date_dim", "d_date_sk"),
    ("catalog_page", "cp_end_date_sk", "date_dim", "d_date_sk"),
    # web_page
    ("web_page", "wp_creation_date_sk", "date_dim", "d_date_sk"),
    ("web_page", "wp_access_date_sk", "date_dim", "d_date_sk"),
    # web_site
    ("web_site", "web_open_date_sk", "date_dim", "d_date_sk"),
    ("web_site", "web_close_date_sk", "date_dim", "d_date_sk"),
    # promotion
    ("promotion", "p_start_date_sk", "date_dim", "d_date_sk"),
    ("promotion", "p_end_date_sk", "date_dim", "d_date_sk"),
    ("promotion", "p_item_sk", "item", "i_item_sk"),
]


def build_schema_map(con) -> Dict[str, List[Tuple[str, str]]]:
    schema_map = {}
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()
    for (table,) in tables:
        cols = con.execute(f"PRAGMA table_info('{table}')").fetchall()
        schema_map[table] = [(c[1], c[2]) for c in cols]
    return schema_map


def filter_foreign_keys(schema_map: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    fks_by_table: Dict[str, List[Dict[str, str]]] = {t: [] for t in schema_map}
    col_lookup = {t: {c for c, _ in cols} for t, cols in schema_map.items()}

    for src_table, src_col, ref_table, ref_col in FOREIGN_KEY_CANDIDATES:
        if src_table not in schema_map or ref_table not in schema_map:
            continue
        if src_col not in col_lookup[src_table] or ref_col not in col_lookup[ref_table]:
            continue
        fks_by_table[src_table].append(
            {"column": src_col, "references": f"{ref_table}.{ref_col}"}
        )
    return fks_by_table


def is_string_type(type_str: str) -> bool:
    upper = type_str.upper()
    return "CHAR" in upper or "VARCHAR" in upper or "TEXT" in upper


def is_date_type(type_str: str) -> bool:
    upper = type_str.upper()
    return "DATE" in upper or "TIME" in upper


def is_numeric_type(type_str: str) -> bool:
    upper = type_str.upper()
    return any(tok in upper for tok in ["INT", "DECIMAL", "NUMERIC", "DOUBLE", "FLOAT", "REAL"])


def build_schema_info(
    table: str,
    schema_map: Dict[str, List[Tuple[str, str]]],
    foreign_keys: Dict[str, List[Dict[str, str]]],
) -> Dict[str, object]:
    cols = schema_map[table]
    features = [{"name": col, "type": typ} for col, typ in cols]
    pk_cols = [c for c in PRIMARY_KEYS.get(table, []) if c in {col for col, _ in cols}]
    return {
        "name": table,
        "purpose": TABLE_PURPOSES.get(table, f"TPC-DS table: {table}."),
        "features": features,
        "keys": {
            "primary_key": pk_cols,
            "foreign_keys": foreign_keys.get(table, []),
        },
    }


def generate_samples_for_table(
    table: str,
    cols: List[Tuple[str, str]],
    samples_per_schema: int,
) -> List[Dict[str, str]]:
    columns = [c for c, _ in cols]
    types = {c: t for c, t in cols}

    string_cols = [c for c in columns if is_string_type(types[c])]
    date_cols = [c for c in columns if is_date_type(types[c])]
    numeric_cols = [c for c in columns if is_numeric_type(types[c])]
    measure_cols = [c for c in numeric_cols if not c.endswith("_sk") and not c.endswith("_id")]

    def pick_cols(n: int) -> List[str]:
        return columns[:n] if len(columns) >= n else columns[:]

    samples: List[Dict[str, str]] = []
    seen_sql = set()

    def add_sample(question: str, sql: str) -> None:
        if sql in seen_sql:
            return
        seen_sql.add(sql)
        samples.append({"question": question, "sql": sql})

    preview_cols = pick_cols(3)
    if preview_cols:
        cols_list = ", ".join(preview_cols)
        add_sample(
            f"Show 5 rows from {table} with columns {cols_list}.",
            f"SELECT {cols_list} FROM {table} LIMIT 5;",
        )

    add_sample(
        f"How many rows are in {table}?",
        f"SELECT COUNT(*) AS row_count FROM {table};",
    )

    if string_cols:
        col = string_cols[0]
        add_sample(
            f"List distinct values of {col} from {table} (limit 10).",
            f"SELECT DISTINCT {col} FROM {table} LIMIT 10;",
        )

    if columns:
        col = columns[0]
        cols_list = ", ".join(preview_cols or [col])
        add_sample(
            f"List rows from {table} where {col} is not null.",
            f"SELECT {cols_list} FROM {table} WHERE {col} IS NOT NULL LIMIT 10;",
        )

    order_col = None
    if date_cols:
        order_col = date_cols[0]
    elif measure_cols:
        order_col = measure_cols[0]
    elif numeric_cols:
        order_col = numeric_cols[0]
    elif string_cols:
        order_col = string_cols[0]
    elif columns:
        order_col = columns[0]

    if order_col:
        select_cols = [order_col] + [c for c in columns if c != order_col]
        cols_list = ", ".join(select_cols[:3])
        add_sample(
            f"Show the top 10 rows from {table} ordered by {order_col} descending.",
            f"SELECT {cols_list} FROM {table} ORDER BY {order_col} DESC LIMIT 10;",
        )

    if measure_cols or numeric_cols:
        col = measure_cols[0] if measure_cols else numeric_cols[0]
        add_sample(
            f"Compute the total {col} in {table}.",
            f"SELECT SUM({col}) AS total_{col} FROM {table};",
        )
        add_sample(
            f"Compute the average {col} in {table}.",
            f"SELECT AVG({col}) AS avg_{col} FROM {table};",
        )

    if date_cols:
        col = date_cols[0]
        add_sample(
            f"Get the minimum and maximum {col} in {table}.",
            f"SELECT MIN({col}) AS min_{col}, MAX({col}) AS max_{col} FROM {table};",
        )

    idx = 0
    while len(samples) < max(samples_per_schema, 5):
        col = columns[idx % len(columns)]
        add_sample(
            f"Show 10 rows from {table} with column {col}.",
            f"SELECT {col} FROM {table} LIMIT 10;",
        )
        idx += 1

    return samples[:samples_per_schema]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate schema-aware finetune data for TPC-DS")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output JSONL path")
    parser.add_argument(
        "--samples-per-schema",
        type=int,
        default=6,
        help="Number of samples per schema (5-10 recommended)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_schema < 5 or args.samples_per_schema > 10:
        raise SystemExit("--samples-per-schema must be between 5 and 10")

    con = duckdb.connect(args.db_path)
    schema_map = build_schema_map(con)
    foreign_keys = filter_foreign_keys(schema_map)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    with output_path.open("w", encoding="utf-8") as f:
        for table in sorted(schema_map.keys()):
            schema_info = build_schema_info(table, schema_map, foreign_keys)
            samples = generate_samples_for_table(table, schema_map[table], args.samples_per_schema)
            for sample in samples:
                record = {
                    "schema": schema_info,
                    "question": sample["question"],
                    "sql": sample["sql"],
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                total_samples += 1

    print(f"Wrote {total_samples} samples to {output_path}")


if __name__ == "__main__":
    main()
