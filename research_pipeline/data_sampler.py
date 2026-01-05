"""
Data Sampler Module
Extracts sample data from database to include in LLM context.
Helps model understand actual data format (e.g., 'CA' not 'California').
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import re

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# ========== CONFIG ==========
CACHE_FILE = Path(__file__).parent / "datasets" / "data_samples.json"

# Columns that have enumerated/categorical values worth sampling
CATEGORICAL_COLUMNS = {
    "customer_address": ["ca_state", "ca_country"],
    "customer_demographics": ["cd_gender", "cd_marital_status", "cd_education_status", "cd_credit_rating"],
    "household_demographics": ["hd_buy_potential"],
    "item": ["i_category", "i_class", "i_brand"],
    "date_dim": ["d_year", "d_moy", "d_qoy", "d_weekend", "d_day_name"],
    "store": ["s_state", "s_city"],
    "warehouse": ["w_state", "w_city"],
    "ship_mode": ["sm_type", "sm_carrier"],
    "reason": ["r_reason_desc"],
    "web_page": ["wp_type"],
}

# Columns with text values that need examples
TEXT_COLUMNS = {
    "item": ["i_item_desc"],
    "customer": ["c_first_name", "c_last_name", "c_birth_country"],
    "promotion": ["p_promo_name"],
}


class DataSampler:
    """Extract and cache sample data from database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.samples: Dict[str, Dict[str, List]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cached samples if available."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.samples = json.load(f)
            except:
                self.samples = {}
    
    def _save_cache(self):
        """Save samples to cache."""
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)
    
    def extract_samples(self, force_refresh: bool = False):
        """Extract sample values from database."""
        if self.samples and not force_refresh:
            return
        
        if not HAS_DUCKDB:
            print("DuckDB not available")
            return
        
        con = duckdb.connect(self.db_path, read_only=True)
        
        # Sample categorical columns
        for table, columns in CATEGORICAL_COLUMNS.items():
            if table not in self.samples:
                self.samples[table] = {}
            
            for col in columns:
                try:
                    # Get distinct values (limit to 20)
                    query = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT 20"
                    result = con.execute(query).fetchall()
                    values = [str(r[0]) for r in result if r[0] is not None]
                    self.samples[table][col] = sorted(values)[:15]
                except Exception as e:
                    print(f"Error sampling {table}.{col}: {e}")
        
        # Sample text columns (just 3 examples each)
        for table, columns in TEXT_COLUMNS.items():
            if table not in self.samples:
                self.samples[table] = {}
            
            for col in columns:
                try:
                    query = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT 5"
                    result = con.execute(query).fetchall()
                    values = [str(r[0])[:50] for r in result if r[0] is not None]
                    self.samples[table][col] = values[:3]
                except Exception as e:
                    print(f"Error sampling {table}.{col}: {e}")
        
        con.close()
        self._save_cache()
        print(f"Extracted samples for {len(self.samples)} tables, saved to {CACHE_FILE}")
    
    def get_relevant_samples(self, question: str, tables: List[str]) -> str:
        """Get sample data relevant to the question and selected tables."""
        if not self.samples:
            return ""
        
        question_lower = question.lower()
        lines = ["DATA SAMPLES (use exact values):"]
        
        for table in tables:
            if table not in self.samples:
                continue
            
            table_samples = self.samples[table]
            relevant = []
            
            for col, values in table_samples.items():
                # Check if column is relevant to question
                if self._is_column_relevant(col, question_lower):
                    if values:
                        sample_str = ", ".join(f"'{v}'" for v in values[:5])
                        relevant.append(f"  {col}: {sample_str}")
            
            if relevant:
                lines.append(f"\n{table}:")
                lines.extend(relevant)
        
        if len(lines) == 1:
            return ""  # No relevant samples found
        
        return "\n".join(lines)
    
    def _is_column_relevant(self, col: str, question: str) -> bool:
        """Check if column is relevant to the question."""
        # Keywords that trigger specific columns
        relevance_map = {
            "ca_state": ["bang", "state", "tiểu bang"],
            "ca_country": ["quốc gia", "country", "nước", "việt nam", "japan", "us"],
            "cd_gender": ["giới", "gender", "nam", "nữ", "male", "female"],
            "cd_marital_status": ["hôn nhân", "marital", "kết hôn", "ly hôn", "độc thân"],
            "cd_education_status": ["học vấn", "education", "đại học", "cao đẳng"],
            "i_category": ["danh mục", "category", "loại"],
            "i_class": ["class", "phân loại"],
            "i_brand": ["thương hiệu", "brand"],
            "d_year": ["năm", "year"],
            "d_moy": ["tháng", "month"],
            "d_weekend": ["cuối tuần", "weekend"],
            "s_state": ["bang", "state", "cửa hàng"],
            "w_state": ["bang", "warehouse", "kho"],
            "sm_type": ["vận chuyển", "ship", "giao hàng"],
            "r_reason_desc": ["lý do", "reason", "trả hàng"],
            "c_birth_country": ["quốc gia", "country", "sinh", "birth"],
        }
        
        keywords = relevance_map.get(col, [])
        return any(kw in question for kw in keywords)


def build_context_with_samples(
    question: str, 
    tables: List[str], 
    schema_map: dict,
    sampler: Optional[DataSampler] = None
) -> str:
    """Build context with schema and sample data."""
    from research_pipeline.context_engineering import build_enhanced_context
    
    # Get enhanced context (schema + JOIN hints)
    context = build_enhanced_context(question, tables, schema_map)
    
    # Add sample data if sampler available
    if sampler:
        samples = sampler.get_relevant_samples(question, tables)
        if samples:
            context = f"{context}\n\n{samples}"
    
    return context


# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract data samples from database")
    parser.add_argument("--db", default="research_pipeline/cache/ecommerce_dw.duckdb",
                        help="Path to DuckDB database")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh cache")
    args = parser.parse_args()
    
    sampler = DataSampler(args.db)
    sampler.extract_samples(force_refresh=args.refresh)
    
    # Print samples
    print("\n=== Sample Data ===")
    for table, columns in sampler.samples.items():
        print(f"\n{table}:")
        for col, values in columns.items():
            print(f"  {col}: {values[:5]}")
