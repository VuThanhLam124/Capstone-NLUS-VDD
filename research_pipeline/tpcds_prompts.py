"""
TPC-DS Optimized Prompts for Text-to-SQL
Contains system prompts and schema templates optimized for TPC-DS schema.
"""

# ========== SYSTEM PROMPTS ==========

SYSTEM_PROMPT_BASIC = """You translate user questions into SQL for DuckDB (TPC-DS). Return only SQL, no markdown."""

SYSTEM_PROMPT_TPCDS = """You are a SQL expert for TPC-DS Data Warehouse on DuckDB.

TPC-DS SCHEMA OVERVIEW:
- 3 Sales Channels: store_sales (ss_*), web_sales (ws_*), catalog_sales (cs_*)
- 3 Return Tables: store_returns (sr_*), web_returns (wr_*), catalog_returns (cr_*)
- Key Dimensions: customer (c_*), item (i_*), date_dim (d_*), store (s_*), warehouse (w_*)

COLUMN NAMING CONVENTION:
- *_sk = Surrogate Key (INTEGER, use for JOINs)
- *_id = Business ID (VARCHAR)
- ss_/ws_/cs_ = Store/Web/Catalog Sales prefixes
- d_year, d_moy, d_qoy = Year, Month of Year, Quarter of Year

COMMON JOINS:
- Sales -> Date: ON ss_sold_date_sk = d_date_sk
- Sales -> Item: ON ss_item_sk = i_item_sk
- Sales -> Customer: ON ss_customer_sk = c_customer_sk
- Customer -> Address: ON c_current_addr_sk = ca_address_sk

DUCKDB SPECIFIC:
- Use date_add() not dateadd()
- Use LIMIT not TOP
- String comparison is case-sensitive

Output ONLY valid SQL. No explanations."""

# ========== SCHEMA DESCRIPTIONS ==========

TABLE_DESCRIPTIONS = {
    "store_sales": "Bán hàng tại cửa hàng vật lý. Cột chính: ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_quantity, ss_net_paid",
    "web_sales": "Bán hàng qua website. Cột chính: ws_sold_date_sk, ws_item_sk, ws_bill_customer_sk, ws_quantity, ws_net_paid",
    "catalog_sales": "Bán hàng qua catalog. Cột chính: cs_sold_date_sk, cs_item_sk, cs_bill_customer_sk, cs_quantity, cs_net_paid",
    "store_returns": "Trả hàng cửa hàng. Cột chính: sr_returned_date_sk, sr_item_sk, sr_customer_sk, sr_return_amt",
    "web_returns": "Trả hàng web. Cột chính: wr_returned_date_sk, wr_item_sk, wr_refunded_customer_sk, wr_return_amt",
    "catalog_returns": "Trả hàng catalog. Cột chính: cr_returned_date_sk, cr_item_sk, cr_refunded_customer_sk, cr_return_amt",
    "customer": "Khách hàng. Cột chính: c_customer_sk, c_customer_id, c_first_name, c_last_name, c_birth_country",
    "customer_address": "Địa chỉ khách hàng. Cột chính: ca_address_sk, ca_state, ca_city, ca_country",
    "customer_demographics": "Thông tin nhân khẩu. Cột chính: cd_demo_sk, cd_gender, cd_marital_status, cd_education_status",
    "item": "Sản phẩm. Cột chính: i_item_sk, i_item_id, i_item_desc, i_category, i_brand, i_current_price",
    "date_dim": "Chiều thời gian. Cột chính: d_date_sk, d_date, d_year, d_moy (month), d_qoy (quarter)",
    "store": "Cửa hàng. Cột chính: s_store_sk, s_store_name, s_state, s_city",
    "warehouse": "Kho hàng. Cột chính: w_warehouse_sk, w_warehouse_name, w_state",
    "promotion": "Khuyến mãi. Cột chính: p_promo_sk, p_promo_name, p_channel_email, p_channel_tv",
    "ship_mode": "Phương thức vận chuyển. Cột chính: sm_ship_mode_sk, sm_type (AIR, GROUND, RAIL, SHIP)",
    "reason": "Lý do trả hàng. Cột chính: r_reason_sk, r_reason_desc",
    "inventory": "Tồn kho. Cột chính: inv_item_sk, inv_warehouse_sk, inv_date_sk, inv_quantity_on_hand",
}

COLUMN_DESCRIPTIONS = {
    # Customer
    "c_customer_sk": "Khóa chính khách hàng (INTEGER)",
    "c_customer_id": "Mã khách hàng (VARCHAR)",
    "c_first_name": "Tên",
    "c_last_name": "Họ",
    "c_birth_country": "Quốc gia sinh (VD: JAPAN, UNITED STATES)",
    "c_current_addr_sk": "FK đến customer_address",
    "c_current_cdemo_sk": "FK đến customer_demographics",
    
    # Item
    "i_item_sk": "Khóa chính sản phẩm (INTEGER)",
    "i_item_id": "Mã sản phẩm (VARCHAR)",
    "i_item_desc": "Mô tả sản phẩm",
    "i_category": "Danh mục (Electronics, Books, Home, Men, Women, Children, Sports, Music, Toys, Jewelry)",
    "i_brand": "Thương hiệu",
    "i_current_price": "Giá hiện tại (DECIMAL)",
    
    # Date
    "d_date_sk": "Khóa chính ngày (INTEGER)",
    "d_date": "Ngày (DATE)",
    "d_year": "Năm (1999-2002)",
    "d_moy": "Tháng trong năm (1-12)",
    "d_qoy": "Quý trong năm (1-4)",
    "d_day_name": "Tên ngày (Monday, Tuesday...)",
    
    # Store Sales
    "ss_sold_date_sk": "FK đến date_dim",
    "ss_item_sk": "FK đến item",
    "ss_customer_sk": "FK đến customer",
    "ss_store_sk": "FK đến store",
    "ss_quantity": "Số lượng mua (INTEGER)",
    "ss_net_paid": "Số tiền thanh toán (DECIMAL)",
    "ss_net_profit": "Lợi nhuận (DECIMAL)",
    
    # Store
    "s_store_sk": "Khóa chính cửa hàng",
    "s_store_name": "Tên cửa hàng",
    "s_state": "Bang (VD: CA, NY, TX)",
    "s_city": "Thành phố",
}

TABLE_RELATIONSHIPS = {
    ("store_sales", "date_dim"): "ss_sold_date_sk = d_date_sk",
    ("store_sales", "item"): "ss_item_sk = i_item_sk",
    ("store_sales", "customer"): "ss_customer_sk = c_customer_sk",
    ("store_sales", "store"): "ss_store_sk = s_store_sk",
    ("web_sales", "date_dim"): "ws_sold_date_sk = d_date_sk",
    ("web_sales", "item"): "ws_item_sk = i_item_sk",
    ("web_sales", "customer"): "ws_bill_customer_sk = c_customer_sk",
    ("catalog_sales", "date_dim"): "cs_sold_date_sk = d_date_sk",
    ("catalog_sales", "item"): "cs_item_sk = i_item_sk",
    ("catalog_sales", "customer"): "cs_bill_customer_sk = c_customer_sk",
    ("customer", "customer_address"): "c_current_addr_sk = ca_address_sk",
    ("customer", "customer_demographics"): "c_current_cdemo_sk = cd_demo_sk",
    ("inventory", "item"): "inv_item_sk = i_item_sk",
    ("inventory", "warehouse"): "inv_warehouse_sk = w_warehouse_sk",
    ("inventory", "date_dim"): "inv_date_sk = d_date_sk",
}

def build_enhanced_schema_text(tables: list, schema_map: dict, include_descriptions: bool = True) -> str:
    """Build schema text with table/column descriptions."""
    lines = []
    
    for table in tables:
        cols = schema_map.get(table, [])
        
        # Table header with description
        desc = TABLE_DESCRIPTIONS.get(table, "")
        if include_descriptions and desc:
            lines.append(f"-- {desc}")
        lines.append(f"TABLE {table} (")
        
        for col, typ in cols:
            col_desc = COLUMN_DESCRIPTIONS.get(col, "")
            if include_descriptions and col_desc:
                lines.append(f"  {col} {typ}, -- {col_desc}")
            else:
                lines.append(f"  {col} {typ},")
        
        lines.append(")")
        lines.append("")
    
    # Add relevant relationships
    if include_descriptions:
        lines.append("-- RELATIONSHIPS:")
        for (t1, t2), join_cond in TABLE_RELATIONSHIPS.items():
            if t1 in tables and t2 in tables:
                lines.append(f"-- {t1} JOIN {t2} ON {join_cond}")
    
    return "\n".join(lines).strip()

def get_system_prompt(mode: str = "tpcds") -> str:
    """Get system prompt by mode."""
    if mode == "basic":
        return SYSTEM_PROMPT_BASIC
    return SYSTEM_PROMPT_TPCDS
