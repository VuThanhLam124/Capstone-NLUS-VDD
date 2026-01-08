"""
Schema Linking Module for Text-to-SQL (TPC-DS)
Implements:
1. Bidirectional Linking: Question→Schema + Schema→Question
2. Vector-based Retrieval: Embedding similarity
3. Context Engineering: Dynamic schema selection
"""
import re
import json
from typing import List, Dict, Set, Tuple
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("WARNING: sentence-transformers not installed for schema linking")

# ========== TPC-DS SCHEMA DEFINITION (ALL 24 TABLES) ==========
TPCDS_TABLES = {
    # === FACT TABLES (6) ===
    "store_sales": {
        "alias": "ss",
        "columns": ["ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk", 
                   "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk", "ss_promo_sk",
                   "ss_ticket_number", "ss_quantity", "ss_wholesale_cost", "ss_list_price",
                   "ss_sales_price", "ss_ext_sales_price", "ss_ext_tax", "ss_net_paid", "ss_net_profit"],
        "keywords": ["store", "sales", "bán hàng", "cửa hàng", "doanh thu", "retail", "thuế", "tax"],
    },
    "store_returns": {
        "alias": "sr",
        "columns": ["sr_returned_date_sk", "sr_return_time_sk", "sr_item_sk", "sr_customer_sk",
                   "sr_cdemo_sk", "sr_hdemo_sk", "sr_addr_sk", "sr_store_sk", "sr_reason_sk",
                   "sr_ticket_number", "sr_return_quantity", "sr_return_amt", "sr_net_loss"],
        "keywords": ["return", "trả hàng", "hoàn trả", "store return", "trả lại"],
    },
    "web_sales": {
        "alias": "ws",
        "columns": ["ws_sold_date_sk", "ws_sold_time_sk", "ws_ship_date_sk", "ws_item_sk", 
                   "ws_bill_customer_sk", "ws_ship_customer_sk", "ws_web_page_sk", "ws_web_site_sk",
                   "ws_ship_mode_sk", "ws_warehouse_sk", "ws_promo_sk", "ws_order_number",
                   "ws_quantity", "ws_sales_price", "ws_net_paid", "ws_net_profit"],
        "keywords": ["web", "online", "internet", "website", "ecommerce", "trực tuyến"],
    },
    "web_returns": {
        "alias": "wr",
        "columns": ["wr_returned_date_sk", "wr_returned_time_sk", "wr_item_sk", 
                   "wr_refunded_customer_sk", "wr_returning_customer_sk", "wr_web_page_sk",
                   "wr_reason_sk", "wr_order_number", "wr_return_quantity", "wr_return_amt", "wr_net_loss"],
        "keywords": ["web return", "online return", "trả hàng online"],
    },
    "catalog_sales": {
        "alias": "cs",
        "columns": ["cs_sold_date_sk", "cs_sold_time_sk", "cs_ship_date_sk", "cs_bill_customer_sk",
                   "cs_ship_customer_sk", "cs_call_center_sk", "cs_catalog_page_sk", "cs_ship_mode_sk",
                   "cs_warehouse_sk", "cs_item_sk", "cs_promo_sk", "cs_order_number",
                   "cs_quantity", "cs_sales_price", "cs_net_paid", "cs_net_profit"],
        "keywords": ["catalog", "catalogue", "danh mục", "mail order"],
    },
    "catalog_returns": {
        "alias": "cr",
        "columns": ["cr_returned_date_sk", "cr_returned_time_sk", "cr_item_sk",
                   "cr_refunded_customer_sk", "cr_returning_customer_sk", "cr_call_center_sk",
                   "cr_catalog_page_sk", "cr_ship_mode_sk", "cr_warehouse_sk", "cr_reason_sk",
                   "cr_order_number", "cr_return_quantity", "cr_return_amount", "cr_net_loss"],
        "keywords": ["catalog return", "trả hàng catalog"],
    },
    "inventory": {
        "alias": "inv",
        "columns": ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk", "inv_quantity_on_hand"],
        "keywords": ["inventory", "stock", "kho", "tồn kho", "warehouse"],
    },
    
    # === DIMENSION TABLES (17) ===
    "customer": {
        "alias": "c",
        "columns": ["c_customer_sk", "c_customer_id", "c_current_cdemo_sk", "c_current_hdemo_sk",
                   "c_current_addr_sk", "c_first_name", "c_last_name", "c_birth_day", "c_birth_month",
                   "c_birth_year", "c_birth_country", "c_login", "c_email_address", "c_preferred_cust_flag"],
        "keywords": ["customer", "khách hàng", "người mua", "buyer", "email", "login"],
    },
    "customer_address": {
        "alias": "ca",
        "columns": ["ca_address_sk", "ca_address_id", "ca_street_number", "ca_street_name", 
                   "ca_street_type", "ca_city", "ca_county", "ca_state", "ca_zip", "ca_country", "ca_gmt_offset"],
        "keywords": ["address", "địa chỉ", "state", "city", "bang", "thành phố", "location"],
    },
    "customer_demographics": {
        "alias": "cd",
        "columns": ["cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status",
                   "cd_purchase_estimate", "cd_credit_rating", "cd_dep_count", "cd_dep_employed_count", "cd_dep_college_count"],
        "keywords": ["gender", "education", "marital", "giới tính", "học vấn", "hôn nhân", "demographics"],
    },
    "household_demographics": {
        "alias": "hd",
        "columns": ["hd_demo_sk", "hd_income_band_sk", "hd_buy_potential", "hd_dep_count", "hd_vehicle_count"],
        "keywords": ["household", "income", "vehicle", "hộ gia đình", "thu nhập", "xe"],
    },
    "income_band": {
        "alias": "ib",
        "columns": ["ib_income_band_sk", "ib_lower_bound", "ib_upper_bound"],
        "keywords": ["income", "salary", "thu nhập", "lương"],
    },
    "item": {
        "alias": "i",
        "columns": ["i_item_sk", "i_item_id", "i_rec_start_date", "i_rec_end_date", "i_item_desc",
                   "i_current_price", "i_wholesale_cost", "i_brand_id", "i_brand", "i_class_id", 
                   "i_class", "i_category_id", "i_category", "i_manufact_id", "i_manufact",
                   "i_size", "i_color", "i_units", "i_container", "i_product_name"],
        "keywords": ["item", "product", "sản phẩm", "brand", "thương hiệu", "category", "hàng hóa"],
    },
    "date_dim": {
        "alias": "d",
        "columns": ["d_date_sk", "d_date_id", "d_date", "d_month_seq", "d_week_seq", "d_quarter_seq",
                   "d_year", "d_dow", "d_moy", "d_dom", "d_qoy", "d_fy_year", "d_day_name", 
                   "d_quarter_name", "d_holiday", "d_weekend"],
        "keywords": ["date", "year", "month", "quarter", "năm", "tháng", "quý", "ngày", "time"],
    },
    "time_dim": {
        "alias": "t",
        "columns": ["t_time_sk", "t_time_id", "t_time", "t_hour", "t_minute", "t_second", 
                   "t_am_pm", "t_shift", "t_sub_shift", "t_meal_time"],
        "keywords": ["time", "hour", "minute", "giờ", "phút", "thời gian"],
    },
    "store": {
        "alias": "s",
        "columns": ["s_store_sk", "s_store_id", "s_store_name", "s_number_employees", "s_floor_space",
                   "s_hours", "s_manager", "s_market_id", "s_geography_class", "s_market_desc",
                   "s_city", "s_county", "s_state", "s_zip", "s_country"],
        "keywords": ["store", "cửa hàng", "shop", "retail store"],
    },
    "warehouse": {
        "alias": "w",
        "columns": ["w_warehouse_sk", "w_warehouse_id", "w_warehouse_name", "w_warehouse_sq_ft",
                   "w_city", "w_county", "w_state", "w_zip", "w_country"],
        "keywords": ["warehouse", "kho hàng", "storage", "distribution"],
    },
    "web_site": {
        "alias": "web",
        "columns": ["web_site_sk", "web_site_id", "web_name", "web_open_date_sk", "web_close_date_sk",
                   "web_class", "web_manager", "web_company_name", "web_city", "web_state"],
        "keywords": ["website", "web site", "trang web", "online store"],
    },
    "web_page": {
        "alias": "wp",
        "columns": ["wp_web_page_sk", "wp_web_page_id", "wp_creation_date_sk", "wp_customer_sk",
                   "wp_url", "wp_type", "wp_char_count", "wp_link_count", "wp_image_count"],
        "keywords": ["web page", "webpage", "page", "trang", "url"],
    },
    "call_center": {
        "alias": "cc",
        "columns": ["cc_call_center_sk", "cc_call_center_id", "cc_name", "cc_class", "cc_employees",
                   "cc_sq_ft", "cc_hours", "cc_manager", "cc_city", "cc_county", "cc_state"],
        "keywords": ["call center", "hotline", "tổng đài", "support center"],
    },
    "catalog_page": {
        "alias": "cp",
        "columns": ["cp_catalog_page_sk", "cp_catalog_page_id", "cp_start_date_sk", "cp_end_date_sk",
                   "cp_department", "cp_catalog_number", "cp_catalog_page_number", "cp_description"],
        "keywords": ["catalog page", "trang catalog", "catalog"],
    },
    "promotion": {
        "alias": "p",
        "columns": ["p_promo_sk", "p_promo_id", "p_start_date_sk", "p_end_date_sk", "p_item_sk",
                   "p_cost", "p_promo_name", "p_channel_dmail", "p_channel_email", "p_channel_tv",
                   "p_channel_radio", "p_discount_active"],
        "keywords": ["promotion", "promo", "discount", "khuyến mãi", "giảm giá", "sale"],
    },
    "reason": {
        "alias": "r",
        "columns": ["r_reason_sk", "r_reason_id", "r_reason_desc"],
        "keywords": ["reason", "lý do", "cause", "return reason"],
    },
    "ship_mode": {
        "alias": "sm",
        "columns": ["sm_ship_mode_sk", "sm_ship_mode_id", "sm_type", "sm_code", "sm_carrier", "sm_contract"],
        "keywords": ["ship", "shipping", "delivery", "vận chuyển", "giao hàng", "carrier"],
    },
}

# Column semantic mappings
COLUMN_SEMANTICS = {
    "revenue": ["ss_net_paid", "ws_net_paid", "cs_net_paid"],
    "doanh thu": ["ss_net_paid", "ws_net_paid", "cs_net_paid"],
    "sales": ["ss_net_paid", "ws_net_paid", "cs_net_paid"],
    "quantity": ["ss_quantity", "ws_quantity", "cs_quantity"],
    "số lượng": ["ss_quantity", "ws_quantity", "cs_quantity"],
    "price": ["i_current_price", "ss_sales_price"],
    "giá": ["i_current_price", "ss_sales_price"],
    "year": ["d_year"],
    "năm": ["d_year"],
    "month": ["d_moy"],
    "tháng": ["d_moy"],
    "state": ["ca_state", "s_state"],
    "bang": ["ca_state", "s_state"],
    "city": ["ca_city", "s_city"],
    "thành phố": ["ca_city", "s_city"],
    "gender": ["cd_gender"],
    "giới tính": ["cd_gender"],
}

# JOIN paths (ALL key relationships)
JOIN_RELATIONSHIPS = {
    # Store Sales
    ("store_sales", "date_dim"): "ss_sold_date_sk = d_date_sk",
    ("store_sales", "item"): "ss_item_sk = i_item_sk",
    ("store_sales", "customer"): "ss_customer_sk = c_customer_sk",
    ("store_sales", "store"): "ss_store_sk = s_store_sk",
    ("store_sales", "promotion"): "ss_promo_sk = p_promo_sk",
    ("store_sales", "customer_demographics"): "ss_cdemo_sk = cd_demo_sk",
    ("store_sales", "household_demographics"): "ss_hdemo_sk = hd_demo_sk",
    ("store_sales", "customer_address"): "ss_addr_sk = ca_address_sk",
    ("store_sales", "time_dim"): "ss_sold_time_sk = t_time_sk",
    
    # Store Returns
    ("store_returns", "date_dim"): "sr_returned_date_sk = d_date_sk",
    ("store_returns", "item"): "sr_item_sk = i_item_sk",
    ("store_returns", "customer"): "sr_customer_sk = c_customer_sk",
    ("store_returns", "store"): "sr_store_sk = s_store_sk",
    ("store_returns", "reason"): "sr_reason_sk = r_reason_sk",
    
    # Web Sales
    ("web_sales", "date_dim"): "ws_sold_date_sk = d_date_sk",
    ("web_sales", "item"): "ws_item_sk = i_item_sk",
    ("web_sales", "customer"): "ws_bill_customer_sk = c_customer_sk",
    ("web_sales", "web_site"): "ws_web_site_sk = web_site_sk",
    ("web_sales", "web_page"): "ws_web_page_sk = wp_web_page_sk",
    ("web_sales", "warehouse"): "ws_warehouse_sk = w_warehouse_sk",
    ("web_sales", "ship_mode"): "ws_ship_mode_sk = sm_ship_mode_sk",
    ("web_sales", "promotion"): "ws_promo_sk = p_promo_sk",
    
    # Web Returns
    ("web_returns", "date_dim"): "wr_returned_date_sk = d_date_sk",
    ("web_returns", "item"): "wr_item_sk = i_item_sk",
    ("web_returns", "web_page"): "wr_web_page_sk = wp_web_page_sk",
    ("web_returns", "reason"): "wr_reason_sk = r_reason_sk",
    
    # Catalog Sales
    ("catalog_sales", "date_dim"): "cs_sold_date_sk = d_date_sk",
    ("catalog_sales", "item"): "cs_item_sk = i_item_sk",
    ("catalog_sales", "customer"): "cs_bill_customer_sk = c_customer_sk",
    ("catalog_sales", "call_center"): "cs_call_center_sk = cc_call_center_sk",
    ("catalog_sales", "catalog_page"): "cs_catalog_page_sk = cp_catalog_page_sk",
    ("catalog_sales", "warehouse"): "cs_warehouse_sk = w_warehouse_sk",
    ("catalog_sales", "ship_mode"): "cs_ship_mode_sk = sm_ship_mode_sk",
    ("catalog_sales", "promotion"): "cs_promo_sk = p_promo_sk",
    
    # Catalog Returns
    ("catalog_returns", "date_dim"): "cr_returned_date_sk = d_date_sk",
    ("catalog_returns", "item"): "cr_item_sk = i_item_sk",
    ("catalog_returns", "call_center"): "cr_call_center_sk = cc_call_center_sk",
    ("catalog_returns", "reason"): "cr_reason_sk = r_reason_sk",
    
    # Inventory
    ("inventory", "date_dim"): "inv_date_sk = d_date_sk",
    ("inventory", "item"): "inv_item_sk = i_item_sk",
    ("inventory", "warehouse"): "inv_warehouse_sk = w_warehouse_sk",
    
    # Customer Dimensions
    ("customer", "customer_address"): "c_current_addr_sk = ca_address_sk",
    ("customer", "customer_demographics"): "c_current_cdemo_sk = cd_demo_sk",
    ("customer", "household_demographics"): "c_current_hdemo_sk = hd_demo_sk",
    
    # Household -> Income Band
    ("household_demographics", "income_band"): "hd_income_band_sk = ib_income_band_sk",
}


class SchemaLinker:
    """
    Bidirectional Schema Linking with Vector Retrieval
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.table_embeddings = {}
        self.column_embeddings = {}
        
        if HAS_EMBEDDINGS:
            self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings for all tables/columns"""
        print(f"Loading schema embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        # Embed tables
        for table_name, table_info in TPCDS_TABLES.items():
            # Table description = name + keywords
            table_desc = f"{table_name} {' '.join(table_info['keywords'])}"
            self.table_embeddings[table_name] = self.model.encode(table_desc, normalize_embeddings=True)
        
        # Embed columns
        all_columns = []
        for table_name, table_info in TPCDS_TABLES.items():
            for col in table_info["columns"]:
                # Column description = table + column
                col_desc = f"{table_name}.{col}"
                all_columns.append(col_desc)
                self.column_embeddings[col] = self.model.encode(col_desc, normalize_embeddings=True)
        
        print(f"Embedded {len(self.table_embeddings)} tables, {len(self.column_embeddings)} columns")
    
    def link_schema(self, question: str, top_k_tables: int = 5, top_k_columns: int = 10) -> Dict:
        """
        Bidirectional Schema Linking
        Returns: {tables: [...], columns: [...], joins: [...]}
        """
        # Forward: Question → Schema (vector retrieval)
        forward_tables, forward_columns = self._forward_linking(question, top_k_tables, top_k_columns)
        
        # Backward: Schema → Question (keyword matching)
        backward_tables, backward_columns = self._backward_linking(question)
        
        # Merge results
        linked_tables = list(set(forward_tables + backward_tables))
        linked_columns = list(set(forward_columns + backward_columns))
        
        # Infer JOIN paths
        join_hints = self._infer_joins(linked_tables)
        
        return {
            "tables": linked_tables[:top_k_tables],
            "columns": linked_columns[:top_k_columns],
            "joins": join_hints,
            "method": "bidirectional"
        }
    
    def _forward_linking(self, question: str, top_k_tables: int, top_k_columns: int) -> Tuple[List, List]:
        """Forward: Question → Schema (semantic similarity)"""
        if not HAS_EMBEDDINGS or self.model is None:
            return [], []
        
        # Embed question
        q_embedding = self.model.encode(question, normalize_embeddings=True)
        
        # Find similar tables
        table_scores = []
        for table_name, table_emb in self.table_embeddings.items():
            score = np.dot(q_embedding, table_emb)
            table_scores.append((table_name, score))
        table_scores.sort(key=lambda x: x[1], reverse=True)
        top_tables = [t[0] for t in table_scores[:top_k_tables]]
        
        # Find similar columns
        column_scores = []
        for col_name, col_emb in self.column_embeddings.items():
            score = np.dot(q_embedding, col_emb)
            column_scores.append((col_name, score))
        column_scores.sort(key=lambda x: x[1], reverse=True)
        top_columns = [c[0] for c in column_scores[:top_k_columns]]
        
        return top_tables, top_columns
    
    def _backward_linking(self, question: str) -> Tuple[List, List]:
        """Backward: Schema → Question (keyword matching)"""
        question_lower = question.lower()
        matched_tables = []
        matched_columns = []
        
        # Match tables by keywords
        for table_name, table_info in TPCDS_TABLES.items():
            for keyword in table_info["keywords"]:
                if keyword.lower() in question_lower:
                    matched_tables.append(table_name)
                    break
        
        # Match columns by semantics
        for semantic_word, column_list in COLUMN_SEMANTICS.items():
            if semantic_word.lower() in question_lower:
                matched_columns.extend(column_list)
        
        return matched_tables, matched_columns
    
    def _infer_joins(self, tables: List[str]) -> List[str]:
        """Infer JOIN conditions from selected tables"""
        join_hints = []
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                key = (table1, table2)
                reverse_key = (table2, table1)
                
                if key in JOIN_RELATIONSHIPS:
                    join_hints.append(f"JOIN {table2} ON {JOIN_RELATIONSHIPS[key]}")
                elif reverse_key in JOIN_RELATIONSHIPS:
                    join_hints.append(f"JOIN {table2} ON {JOIN_RELATIONSHIPS[reverse_key]}")
        
        return join_hints
    
    def build_dynamic_schema(self, question: str, max_tables: int = 4) -> str:
        """
        Build dynamic schema context for question
        Returns schema in SAME FORMAT as training data (multi-line with types)
        """
        linking_result = self.link_schema(question, top_k_tables=max_tables, top_k_columns=15)
        
        # Add ALIAS MAPPING first (prevent alias confusion)
        schema_lines = []
        schema_lines.append("TABLE ALIASES (USE EXACTLY):")
        for table_name in linking_result["tables"]:
            if table_name in TPCDS_TABLES:
                alias = TPCDS_TABLES[table_name]["alias"]
                schema_lines.append(f"  {table_name} = {alias}")
        schema_lines.append("")
        
        # Build schema in TRAINING DATA FORMAT (multi-line with types + FK comments)
        for table_name in linking_result["tables"]:
            if table_name not in TPCDS_TABLES:
                continue
            
            table_info = TPCDS_TABLES[table_name]
            alias = table_info["alias"]
            
            # Start table definition with alias
            schema_lines.append(f"TABLE {table_name} (alias: {alias})")
            
            # Add each column on separate line with type (match training format)
            for col in table_info["columns"]:
                col_type = self._get_column_type(col)
                fk_comment = self._get_fk_comment(table_name, col)
                if fk_comment:
                    schema_lines.append(f"  {col} {col_type}   -- {fk_comment}")
                else:
                    schema_lines.append(f"  {col} {col_type}")
            
            schema_lines.append("")  # Empty line between tables
        
        # Add JOIN hints
        if linking_result["joins"]:
            schema_lines.append("JOIN HINTS:")
            schema_lines.extend([f"  {j}" for j in linking_result["joins"][:3]])
            schema_lines.append("")
        
        # Add COLUMN WARNINGS based on question
        warnings = self._get_column_warnings(question)
        if warnings:
            schema_lines.append("IMPORTANT:")
            schema_lines.extend([f"  - {w}" for w in warnings])
        
        return "\n".join(schema_lines)
    
    def _get_column_type(self, col_name: str) -> str:
        """Get column data type based on naming convention."""
        if col_name.endswith("_sk"):
            return "BIGINT"
        elif col_name.endswith("_id"):
            return "VARCHAR"
        elif col_name.endswith("_date"):
            return "DATE"
        elif col_name.endswith("_name") or col_name.endswith("_desc"):
            return "VARCHAR"
        elif col_name.endswith("_price") or col_name.endswith("_amt") or col_name.endswith("_cost"):
            return "DECIMAL"
        elif col_name.endswith("_count") or col_name.endswith("_quantity"):
            return "INTEGER"
        elif "year" in col_name or "month" in col_name or "day" in col_name:
            return "BIGINT"
        else:
            return "VARCHAR"
    
    def _get_fk_comment(self, table_name: str, col_name: str) -> str:
        """Get FK comment for column."""
        fk_mappings = {
            "c_current_cdemo_sk": "FK -> customer_demographics.cd_demo_sk",
            "c_current_hdemo_sk": "FK -> household_demographics.hd_demo_sk",
            "c_current_addr_sk": "FK -> customer_address.ca_address_sk",
            "ss_sold_date_sk": "FK -> date_dim.d_date_sk",
            "ss_item_sk": "FK -> item.i_item_sk",
            "ss_customer_sk": "FK -> customer.c_customer_sk",
            "ss_store_sk": "FK -> store.s_store_sk",
            "ws_sold_date_sk": "FK -> date_dim.d_date_sk",
            "ws_item_sk": "FK -> item.i_item_sk",
            "ws_bill_customer_sk": "FK -> customer.c_customer_sk",
            "ws_web_page_sk": "FK -> web_page.wp_web_page_sk",
            "cs_sold_date_sk": "FK -> date_dim.d_date_sk",
            "cs_item_sk": "FK -> item.i_item_sk",
            "cs_bill_customer_sk": "FK -> customer.c_customer_sk",
            "inv_date_sk": "FK -> date_dim.d_date_sk",
            "inv_item_sk": "FK -> item.i_item_sk",
            "inv_warehouse_sk": "FK -> warehouse.w_warehouse_sk",
            "hd_income_band_sk": "FK -> income_band.ib_income_band_sk",
        }
        
        # Check for PK
        if col_name.endswith("_sk") and col_name.startswith(table_name[:2]):
            return "PRIMARY KEY"
        
        return fk_mappings.get(col_name, "")
    
    def _get_column_warnings(self, question: str) -> List[str]:
        """Generate warnings about commonly confused columns based on question."""
        question_lower = question.lower()
        warnings = []
        
        # ============ CHANNEL DISAMBIGUATION (CRITICAL!) ============
        # These sales channels are DIFFERENT and should NOT be confused:
        # - store_sales (ss): Physical retail stores
        # - web_sales (ws): Online website sales  
        # - catalog_sales (cs): Mail-order catalog sales
        
        web_keywords = ["web", "online", "website", "internet", "trang web", "web_page", "wp_"]
        catalog_keywords = ["catalog", "catalogue", "danh mục", "mail order", "call_center", "cc_"]
        store_keywords = ["store", "cửa hàng", "retail", "shop", "s_store"]
        
        has_web = any(kw in question_lower for kw in web_keywords)
        has_catalog = any(kw in question_lower for kw in catalog_keywords)
        has_store = any(kw in question_lower for kw in store_keywords)
        
        if has_web:
            warnings.append("CHANNEL: 'web/online' -> USE web_sales (ws), web_returns (wr), web_page (wp)")
            warnings.append("   DO NOT use catalog_sales for web questions!")
        elif has_catalog:
            warnings.append("CHANNEL: 'catalog/mail' -> USE catalog_sales (cs), catalog_returns (cr)")
            warnings.append("   DO NOT use web_sales for catalog questions!")
        elif has_store:
            warnings.append("CHANNEL: 'store/cửa hàng' -> USE store_sales (ss), store_returns (sr)")
        
        # ============ TABLE-COLUMN OWNERSHIP (CRITICAL!) ============
        # These columns are often confused between tables:
        
        # Gender confusion - VERY COMMON ERROR
        if "gender" in question_lower or "giới tính" in question_lower or "nam" in question_lower or "nữ" in question_lower:
            warnings.append("GENDER: cd_gender is in customer_demographics (cd), NOT in customer (c)")
            warnings.append("   Use: JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk")
            warnings.append("   Then: cd.cd_gender (NOT c.c_gender)")
        
        # Marital status confusion - VERY COMMON ERROR  
        if "hôn nhân" in question_lower or "marital" in question_lower or "ly hôn" in question_lower or "divorce" in question_lower:
            warnings.append("MARITAL: cd_marital_status is in customer_demographics (cd), NOT in household_demographics (hd)")
            warnings.append("   Use: JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk")
            warnings.append("   Values: 'S'=Single, 'M'=Married, 'D'=Divorced, 'W'=Widowed")
        
        # Credit rating confusion
        if "credit" in question_lower or "tín dụng" in question_lower:
            warnings.append("CREDIT: cd_credit_rating is in customer_demographics (cd), NOT in customer")
            warnings.append("   Values: 'Low Risk', 'Good', 'High Risk', etc.")
        
        # Vehicle count confusion
        if "vehicle" in question_lower or "xe" in question_lower or "ô tô" in question_lower:
            warnings.append("VEHICLE: hd_vehicle_count is in household_demographics (hd), NOT in customer")
            warnings.append("   Use: JOIN household_demographics hd ON c.c_current_hdemo_sk = hd.hd_demo_sk")
        
        # Dependent count confusion
        if "phụ thuộc" in question_lower or "dependent" in question_lower:
            warnings.append("DEPENDENTS: cd_dep_count (customer_demographics) or hd_dep_count (household)")
            
        # ============ COMMON SYNTAX ERRORS ============
        warnings.append("SYNTAX: DO NOT double prefix columns (sr_return_amt, NOT sr_sr_return_amt)")
        warnings.append("DATE: Quarter=d_qoy, Day name=d_day_name, Weekend=d_weekend='Y'")
        
        # State confusion
        if "state" in question_lower or "bang" in question_lower:
            warnings.append("STATE: ca_state (customer_address) vs s_state (store) vs w_state (warehouse)")
        
        # URL confusion
        if "url" in question_lower:
            warnings.append("URL: wp_url is in web_page (wp), NOT in web_sales")
        
        # Inventory confusion
        if "tồn kho" in question_lower or "inventory" in question_lower or "kho" in question_lower:
            warnings.append("INVENTORY: Use inv_quantity_on_hand from inventory (inv)")
            warnings.append("   JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk for warehouse info")
        
        # Year/date confusion
        if "năm" in question_lower or "year" in question_lower or "quý" in question_lower or "quarter" in question_lower:
            warnings.append("DATE: Always JOIN date_dim d ON *_sold_date_sk = d.d_date_sk to filter by d_year")
        
        return warnings


# ========== STANDALONE USAGE ==========
if __name__ == "__main__":
    linker = SchemaLinker()
    
    # Test cases
    test_questions = [
        "Năm 2002 thì kênh Store mang về bao nhiêu tiền?",
        "Top 5 sản phẩm bán chạy nhất theo số lượng",
        "Tổng doanh thu theo từng bang của khách hàng",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")
        
        result = linker.link_schema(q, top_k_tables=3, top_k_columns=8)
        print(f"Linked Tables: {result['tables']}")
        print(f"Linked Columns: {result['columns']}")
        print(f"JOIN Hints: {result['joins']}")
        
        print(f"\nDynamic Schema:")
        print(linker.build_dynamic_schema(q, max_tables=3))
