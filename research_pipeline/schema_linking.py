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

# ========== TPC-DS SCHEMA DEFINITION ==========
TPCDS_TABLES = {
    "store_sales": {
        "alias": "ss",
        "columns": ["ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk", 
                   "ss_quantity", "ss_sales_price", "ss_net_paid", "ss_net_profit"],
        "keywords": ["store", "sales", "bán hàng", "cửa hàng", "doanh thu"],
    },
    "store_returns": {
        "alias": "sr",
        "columns": ["sr_returned_date_sk", "sr_item_sk", "sr_customer_sk", 
                   "sr_return_quantity", "sr_return_amt", "sr_net_loss"],
        "keywords": ["return", "trả hàng", "hoàn trả"],
    },
    "web_sales": {
        "alias": "ws",
        "columns": ["ws_sold_date_sk", "ws_item_sk", "ws_bill_customer_sk",
                   "ws_quantity", "ws_sales_price", "ws_net_paid", "ws_net_profit"],
        "keywords": ["web", "online", "internet", "website"],
    },
    "catalog_sales": {
        "alias": "cs",
        "columns": ["cs_sold_date_sk", "cs_item_sk", "cs_bill_customer_sk",
                   "cs_quantity", "cs_sales_price", "cs_net_paid", "cs_net_profit"],
        "keywords": ["catalog", "catalogue", "danh mục"],
    },
    "customer": {
        "alias": "c",
        "columns": ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name",
                   "c_birth_year", "c_email_address"],
        "keywords": ["customer", "khách hàng", "người mua"],
    },
    "customer_address": {
        "alias": "ca",
        "columns": ["ca_address_sk", "ca_street_name", "ca_city", "ca_state", "ca_zip"],
        "keywords": ["address", "địa chỉ", "state", "city", "bang", "thành phố"],
    },
    "customer_demographics": {
        "alias": "cd",
        "columns": ["cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status",
                   "cd_dep_count", "cd_credit_rating"],
        "keywords": ["gender", "education", "marital", "giới tính", "học vấn", "hôn nhân"],
    },
    "item": {
        "alias": "i",
        "columns": ["i_item_sk", "i_item_id", "i_item_desc", "i_current_price",
                   "i_brand", "i_category", "i_class", "i_product_name"],
        "keywords": ["item", "product", "sản phẩm", "brand", "thương hiệu", "category"],
    },
    "date_dim": {
        "alias": "d",
        "columns": ["d_date_sk", "d_date", "d_year", "d_moy", "d_qoy", "d_day_name",
                   "d_weekend", "d_quarter_name"],
        "keywords": ["date", "year", "month", "quarter", "năm", "tháng", "quý", "ngày"],
    },
    "store": {
        "alias": "s",
        "columns": ["s_store_sk", "s_store_id", "s_store_name", "s_city", "s_state",
                   "s_number_employees", "s_floor_space"],
        "keywords": ["store", "cửa hàng", "shop"],
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

# JOIN paths
JOIN_RELATIONSHIPS = {
    ("store_sales", "date_dim"): "ss_sold_date_sk = d_date_sk",
    ("store_sales", "item"): "ss_item_sk = i_item_sk",
    ("store_sales", "customer"): "ss_customer_sk = c_customer_sk",
    ("store_sales", "store"): "ss_store_sk = s_store_sk",
    ("web_sales", "date_dim"): "ws_sold_date_sk = d_date_sk",
    ("web_sales", "item"): "ws_item_sk = i_item_sk",
    ("catalog_sales", "date_dim"): "cs_sold_date_sk = d_date_sk",
    ("customer", "customer_address"): "c_current_addr_sk = ca_address_sk",
    ("customer", "customer_demographics"): "c_current_cdemo_sk = cd_demo_sk",
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
        Returns minimal schema string
        """
        linking_result = self.link_schema(question, top_k_tables=max_tables, top_k_columns=15)
        
        # Build compact schema
        schema_lines = []
        for table_name in linking_result["tables"]:
            if table_name not in TPCDS_TABLES:
                continue
            
            table_info = TPCDS_TABLES[table_name]
            alias = table_info["alias"]
            
            # Filter relevant columns
            relevant_cols = [c for c in table_info["columns"] if c in linking_result["columns"]]
            if not relevant_cols:
                relevant_cols = table_info["columns"][:5]  # Fallback to first 5
            
            cols_str = ", ".join(relevant_cols)
            schema_lines.append(f"TABLE {table_name} ({cols_str})")
        
        # Add JOIN hints
        if linking_result["joins"]:
            schema_lines.append("\nJOIN HINTS:")
            schema_lines.extend([f"  {j}" for j in linking_result["joins"][:3]])
        
        return "\n".join(schema_lines)


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
