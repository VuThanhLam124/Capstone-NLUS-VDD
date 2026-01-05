"""
Context Engineering Module for Text-to-SQL
Implements:
1. Dynamic Context Selection - only relevant columns
2. Relationship Hints - JOIN paths between tables
3. Column Name Corrections - highlight commonly confused columns
"""
import re
from typing import Dict, List, Set, Tuple

# ========== COLUMN RELEVANCE KEYWORDS ==========
# Map keywords to relevant columns
KEYWORD_TO_COLUMNS = {
    # Revenue/Sales
    "doanh thu": ["ss_net_paid", "ws_net_paid", "cs_net_paid", "ss_sales_price", "ws_sales_price", "cs_sales_price"],
    "revenue": ["ss_net_paid", "ws_net_paid", "cs_net_paid"],
    "sales": ["ss_net_paid", "ws_net_paid", "cs_net_paid", "ss_quantity", "ws_quantity"],
    
    # Time
    "năm": ["d_year", "d_date_sk"],
    "year": ["d_year", "d_date_sk"],
    "tháng": ["d_moy", "d_month_seq"],
    "month": ["d_moy", "d_month_seq"],
    "quý": ["d_qoy", "d_quarter_seq"],
    "quarter": ["d_qoy"],
    "ngày": ["d_date", "d_date_sk"],
    "cuối tuần": ["d_weekend"],
    "weekend": ["d_weekend"],
    
    # Customer
    "khách": ["c_customer_sk", "c_first_name", "c_last_name", "c_customer_id"],
    "customer": ["c_customer_sk", "c_first_name", "c_last_name"],
    "giới tính": ["cd_gender"],
    "gender": ["cd_gender"],
    "tuổi": ["cd_birth_year"],
    "age": ["cd_birth_year"],
    "học vấn": ["cd_education_status"],
    "education": ["cd_education_status"],
    "hôn nhân": ["cd_marital_status"],
    "marital": ["cd_marital_status"],
    "ly hôn": ["cd_marital_status"],
    "đã kết hôn": ["cd_marital_status"],
    "phụ thuộc": ["cd_dep_count", "hd_dep_count"],
    "dependent": ["cd_dep_count", "hd_dep_count"],
    
    # Location
    "bang": ["ca_state", "s_state"],
    "state": ["ca_state", "s_state"],
    "thành phố": ["ca_city", "s_city"],
    "city": ["ca_city", "s_city"],
    "địa chỉ": ["ca_address_sk", "ca_street_name"],
    
    # Product
    "sản phẩm": ["i_item_sk", "i_item_id", "i_item_desc", "i_category"],
    "item": ["i_item_sk", "i_item_id", "i_category"],
    "danh mục": ["i_category", "i_class"],
    "category": ["i_category"],
    "thương hiệu": ["i_brand"],
    "brand": ["i_brand"],
    "giá": ["i_current_price", "ss_sales_price"],
    "price": ["i_current_price", "ss_sales_price"],
    
    # Returns
    "trả hàng": ["sr_return_amt", "sr_return_quantity", "wr_return_amt", "cr_return_amt"],
    "return": ["sr_return_amt", "sr_return_quantity"],
    "hoàn": ["sr_return_amt", "wr_return_amt"],
    
    # Inventory
    "tồn kho": ["inv_quantity_on_hand"],
    "inventory": ["inv_quantity_on_hand"],
    
    # Store
    "cửa hàng": ["s_store_sk", "s_store_name"],
    "store": ["s_store_sk", "s_store_name"],
    
    # Web
    "web": ["ws_net_paid", "wp_url", "web_site_sk"],
    "url": ["wp_url"],
    "trang web": ["wp_web_page_sk", "wp_url"],
    
    # Catalog
    "catalog": ["cs_net_paid", "cp_catalog_page_sk"],
}

# ========== JOIN RELATIONSHIPS ==========
# Define how tables connect
JOIN_PATHS = {
    ("store_sales", "date_dim"): "ss_sold_date_sk = d_date_sk",
    ("store_sales", "item"): "ss_item_sk = i_item_sk",
    ("store_sales", "customer"): "ss_customer_sk = c_customer_sk",
    ("store_sales", "store"): "ss_store_sk = s_store_sk",
    ("store_sales", "promotion"): "ss_promo_sk = p_promo_sk",
    
    ("web_sales", "date_dim"): "ws_sold_date_sk = d_date_sk",
    ("web_sales", "item"): "ws_item_sk = i_item_sk",
    ("web_sales", "customer"): "ws_bill_customer_sk = c_customer_sk",
    ("web_sales", "web_site"): "ws_web_site_sk = web_site_sk",
    ("web_sales", "web_page"): "ws_web_page_sk = wp_web_page_sk",
    
    ("catalog_sales", "date_dim"): "cs_sold_date_sk = d_date_sk",
    ("catalog_sales", "item"): "cs_item_sk = i_item_sk",
    ("catalog_sales", "customer"): "cs_bill_customer_sk = c_customer_sk",
    ("catalog_sales", "catalog_page"): "cs_catalog_page_sk = cp_catalog_page_sk",
    
    ("store_returns", "date_dim"): "sr_returned_date_sk = d_date_sk",
    ("store_returns", "item"): "sr_item_sk = i_item_sk",
    ("store_returns", "customer"): "sr_customer_sk = c_customer_sk",
    ("store_returns", "store"): "sr_store_sk = s_store_sk",
    ("store_returns", "reason"): "sr_reason_sk = r_reason_sk",
    
    ("web_returns", "date_dim"): "wr_returned_date_sk = d_date_sk",
    ("web_returns", "item"): "wr_item_sk = i_item_sk",
    ("web_returns", "customer"): "wr_refunded_customer_sk = c_customer_sk",
    ("web_returns", "reason"): "wr_reason_sk = r_reason_sk",
    
    ("catalog_returns", "date_dim"): "cr_returned_date_sk = d_date_sk",
    ("catalog_returns", "item"): "cr_item_sk = i_item_sk",
    ("catalog_returns", "customer"): "cr_refunded_customer_sk = c_customer_sk",
    ("catalog_returns", "reason"): "cr_reason_sk = r_reason_sk",
    
    ("inventory", "date_dim"): "inv_date_sk = d_date_sk",
    ("inventory", "item"): "inv_item_sk = i_item_sk",
    ("inventory", "warehouse"): "inv_warehouse_sk = w_warehouse_sk",
    
    ("customer", "customer_address"): "c_current_addr_sk = ca_address_sk",
    ("customer", "customer_demographics"): "c_current_cdemo_sk = cd_demo_sk",
    ("customer", "household_demographics"): "c_current_hdemo_sk = hd_demo_sk",
}

# ========== COMMONLY CONFUSED COLUMNS ==========
COLUMN_CORRECTIONS = {
    # Wrong -> Correct
    "cd_customer_sk": "cd_demo_sk (customer_demographics uses cd_demo_sk as PK)",
    "d_state": "ca_state (state is in customer_address, not date_dim)",
    "d_weekday": "d_weekend (use 'Y'/'N' for weekend check)",
    "cs_url": "wp_url (URL is in web_page table)",
    "cd_num_children": "hd_dep_count (children/dependents in household_demographics)",
    "i_item_name": "i_item_desc (TPC-DS uses i_item_desc, not i_item_name)",
}

# ========== ESSENTIAL COLUMNS PER TABLE ==========
ESSENTIAL_COLUMNS = {
    "store_sales": ["ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", "ss_store_sk", "ss_quantity", "ss_net_paid", "ss_sales_price"],
    "web_sales": ["ws_sold_date_sk", "ws_item_sk", "ws_bill_customer_sk", "ws_web_page_sk", "ws_quantity", "ws_net_paid"],
    "catalog_sales": ["cs_sold_date_sk", "cs_item_sk", "cs_bill_customer_sk", "cs_quantity", "cs_net_paid"],
    "store_returns": ["sr_returned_date_sk", "sr_item_sk", "sr_customer_sk", "sr_return_quantity", "sr_return_amt"],
    "web_returns": ["wr_returned_date_sk", "wr_item_sk", "wr_refunded_customer_sk", "wr_return_amt"],
    "catalog_returns": ["cr_returned_date_sk", "cr_item_sk", "cr_refunded_customer_sk", "cr_return_amt"],
    "customer": ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name", "c_current_addr_sk", "c_current_cdemo_sk"],
    "customer_address": ["ca_address_sk", "ca_city", "ca_state", "ca_zip", "ca_country"],
    "customer_demographics": ["cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status", "cd_dep_count"],
    "household_demographics": ["hd_demo_sk", "hd_income_band_sk", "hd_buy_potential", "hd_dep_count", "hd_vehicle_count"],
    "item": ["i_item_sk", "i_item_id", "i_item_desc", "i_category", "i_class", "i_brand", "i_current_price"],
    "date_dim": ["d_date_sk", "d_date", "d_year", "d_moy", "d_qoy", "d_weekend", "d_day_name"],
    "store": ["s_store_sk", "s_store_id", "s_store_name", "s_city", "s_state"],
    "warehouse": ["w_warehouse_sk", "w_warehouse_id", "w_warehouse_name", "w_city", "w_state"],
    "web_page": ["wp_web_page_sk", "wp_web_page_id", "wp_url", "wp_type"],
    "web_site": ["web_site_sk", "web_site_id", "web_name"],
    "catalog_page": ["cp_catalog_page_sk", "cp_catalog_page_id", "cp_department"],
    "promotion": ["p_promo_sk", "p_promo_id", "p_promo_name"],
    "reason": ["r_reason_sk", "r_reason_id", "r_reason_desc"],
    "inventory": ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk", "inv_quantity_on_hand"],
}


def extract_keywords(question: str) -> Set[str]:
    """Extract relevant keywords from question."""
    question_lower = question.lower()
    found_keywords = set()
    
    for keyword in KEYWORD_TO_COLUMNS.keys():
        if keyword in question_lower:
            found_keywords.add(keyword)
    
    return found_keywords


def get_relevant_columns(question: str, tables: List[str], schema_map: dict) -> Dict[str, List[str]]:
    """Get only relevant columns for each table based on question keywords."""
    keywords = extract_keywords(question)
    
    # Get columns suggested by keywords
    suggested_columns = set()
    for kw in keywords:
        suggested_columns.update(KEYWORD_TO_COLUMNS.get(kw, []))
    
    result = {}
    for table in tables:
        if table not in schema_map:
            continue
        
        all_cols = [col for col, _ in schema_map[table]]
        essential = ESSENTIAL_COLUMNS.get(table, [])
        
        # Start with essential columns
        relevant = set(essential)
        
        # Add columns matching keywords
        for col in all_cols:
            if col in suggested_columns:
                relevant.add(col)
        
        # Limit to available columns
        result[table] = [c for c in all_cols if c in relevant]
    
    return result


def get_join_hints(tables: List[str]) -> List[str]:
    """Generate JOIN hints for selected tables."""
    hints = []
    
    for i, t1 in enumerate(tables):
        for t2 in tables[i+1:]:
            # Check both orderings
            key1 = (t1, t2)
            key2 = (t2, t1)
            
            if key1 in JOIN_PATHS:
                hints.append(f"  {t1} JOIN {t2} ON {JOIN_PATHS[key1]}")
            elif key2 in JOIN_PATHS:
                hints.append(f"  {t2} JOIN {t1} ON {JOIN_PATHS[key2]}")
    
    return hints


def get_column_warnings(question: str) -> List[str]:
    """Generate warnings about commonly confused columns."""
    question_lower = question.lower()
    warnings = []
    
    # Check for patterns that might cause confusion
    if "state" in question_lower or "bang" in question_lower:
        warnings.append("NOTE: State is in ca_state (customer_address), NOT in date_dim")
    
    if "gender" in question_lower or "giới" in question_lower:
        warnings.append("NOTE: Gender is in cd_gender (customer_demographics)")
    
    if "weekend" in question_lower or "cuối tuần" in question_lower:
        warnings.append("NOTE: Use d_weekend = 'Y' for weekends")
    
    if "url" in question_lower:
        warnings.append("NOTE: URL is in wp_url (web_page table)")
    
    if "child" in question_lower or "phụ thuộc" in question_lower or "dependent" in question_lower:
        warnings.append("NOTE: Dependents in hd_dep_count (household_demographics) or cd_dep_count")
    
    return warnings


def build_dynamic_schema(tables: List[str], schema_map: dict, question: str) -> str:
    """Build dynamic schema with only relevant columns."""
    relevant_cols = get_relevant_columns(question, tables, schema_map)
    
    lines = []
    for table in tables:
        cols = relevant_cols.get(table, [])
        if not cols:
            # Fallback to essential columns
            cols = ESSENTIAL_COLUMNS.get(table, [])[:8]
        
        # Get column types from schema_map
        col_types = {c: t for c, t in schema_map.get(table, [])}
        
        lines.append(f"TABLE {table} (")
        for col in cols[:12]:  # Limit to 12 columns per table
            typ = col_types.get(col, "")
            lines.append(f"  {col} {typ}")
        lines.append(")")
        lines.append("")
    
    return "\n".join(lines).strip()


def build_enhanced_context(question: str, tables: List[str], schema_map: dict) -> str:
    """Build enhanced context with dynamic schema, JOIN hints, and warnings."""
    parts = []
    
    # 1. Dynamic Schema (only relevant columns)
    schema = build_dynamic_schema(tables, schema_map, question)
    parts.append("SCHEMA:")
    parts.append(schema)
    
    # 2. JOIN Hints
    join_hints = get_join_hints(tables)
    if join_hints:
        parts.append("\nJOIN HINTS:")
        parts.extend(join_hints)
    
    # 3. Column Warnings
    warnings = get_column_warnings(question)
    if warnings:
        parts.append("\nIMPORTANT:")
        parts.extend(warnings)
    
    return "\n".join(parts)


# ========== TEST ==========
if __name__ == "__main__":
    # Test with sample question
    question = "Tổng doanh thu của Music tại bang NC năm 2002"
    tables = ["store_sales", "customer", "customer_address", "item", "date_dim"]
    
    # Mock schema_map
    schema_map = {
        "store_sales": [("ss_sold_date_sk", "INT"), ("ss_item_sk", "INT"), ("ss_customer_sk", "INT"), ("ss_net_paid", "DECIMAL")],
        "customer": [("c_customer_sk", "INT"), ("c_first_name", "VARCHAR"), ("c_current_addr_sk", "INT")],
        "customer_address": [("ca_address_sk", "INT"), ("ca_state", "VARCHAR"), ("ca_city", "VARCHAR")],
        "item": [("i_item_sk", "INT"), ("i_category", "VARCHAR"), ("i_brand", "VARCHAR")],
        "date_dim": [("d_date_sk", "INT"), ("d_year", "INT"), ("d_moy", "INT")],
    }
    
    context = build_enhanced_context(question, tables, schema_map)
    print(context)
