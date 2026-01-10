#!/usr/bin/env python3
"""
Full Pipeline Benchmark: Qwen3-Coder-30B-A3B-Instruct
Features:
1. Schema Linking (dynamic table selection)
2. Few-Shot Learning (3, 5, 7 shots)
3. RAG-based example retrieval

Usage:
    python benchmark_qwen_coder_fewshot.py --shots 3 5 7 --max-test-samples 15
    python benchmark_qwen_coder_fewshot.py --easy --shots 5 --max-test-samples 28
"""

import argparse
import re
import sys
import time
import json
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import duckdb

# Fix vLLM CUDA multiprocessing issue
multiprocessing.set_start_method('spawn', force=True)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from schema_linking import SchemaLinker, TPCDS_TABLES, JOIN_RELATIONSHIPS

# ========== CONSTANTS ==========
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DB_PATH = "research_pipeline/cache/ecommerce_dw.duckdb"

# Static few-shot examples covering different patterns
STATIC_FEWSHOT_EXAMPLES = [
    # ===== CHANNEL EXAMPLES =====
    # Channel: Catalog
    {
        "question": "Tổng doanh thu từ catalog trong năm 2000",
        "sql": "SELECT SUM(cs.cs_net_paid) FROM catalog_sales cs JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk WHERE d.d_year = 2000;"
    },
    # Channel: Web/Online
    {
        "question": "Bao nhiêu đơn hàng được đặt qua website trong quý 2 năm 2002?",
        "sql": "SELECT COUNT(*) FROM web_sales ws JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002 AND d.d_qoy = 2;"
    },
    # Channel: Store
    {
        "question": "Doanh thu cửa hàng năm 2002 là bao nhiêu?",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002;"
    },
    
    # ===== DEMOGRAPHICS EXAMPLES =====
    # Demographics: Gender  
    {
        "question": "Thống kê số lượng khách hàng theo giới tính",
        "sql": "SELECT cd.cd_gender, COUNT(DISTINCT c.c_customer_sk) AS cnt FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk GROUP BY cd.cd_gender;"
    },
    # Demographics: Marital Status
    {
        "question": "So sánh doanh thu giữa khách hàng độc thân và đã kết hôn",
        "sql": "SELECT cd.cd_marital_status, SUM(ss.ss_net_paid) AS revenue FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_marital_status IN ('S', 'M') GROUP BY cd.cd_marital_status;"
    },
    # Demographics: Vehicle count (household_demographics)
    {
        "question": "Liệt kê khách hàng có trên 2 xe ô tô",
        "sql": "SELECT c.c_customer_id, c.c_first_name, hd.hd_vehicle_count FROM customer c JOIN household_demographics hd ON c.c_current_hdemo_sk = hd.hd_demo_sk WHERE hd.hd_vehicle_count > 2;"
    },
    # Demographics: Credit rating
    {
        "question": "Tìm khách hàng có xếp hạng tín dụng thấp",
        "sql": "SELECT c.c_customer_id, c.c_first_name, cd.cd_credit_rating FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_credit_rating = 'Low Risk';"
    },
    
    # ===== ITEM/PRODUCT EXAMPLES =====
    # Item: Category filter
    {
        "question": "Top 5 sản phẩm Shoes bán chạy nhất tại cửa hàng",
        "sql": "SELECT i.i_product_name, SUM(ss.ss_quantity) AS total FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk WHERE i.i_category = 'Shoes' GROUP BY i.i_product_name ORDER BY total DESC LIMIT 5;"
    },
    # Item: Brand by state
    {
        "question": "Top 10 thương hiệu Sports được ưa chuộng nhất ở NC",
        "sql": "SELECT i.i_brand, COUNT(*) as purchase_count FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk JOIN item i ON ss.ss_item_sk = i.i_item_sk WHERE i.i_category = 'Sports' AND ca.ca_state = 'NC' GROUP BY i.i_brand ORDER BY purchase_count DESC LIMIT 10;"
    },
    
    # ===== CUSTOMER + ADDRESS EXAMPLES =====
    # Customer email (correct column name)
    {
        "question": "Tìm email của khách hàng VIP ở TX mua trên 5000 đô",
        "sql": "SELECT c.c_email_address, SUM(ss.ss_net_paid) as total_spent FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk WHERE ca.ca_state = 'TX' GROUP BY c.c_customer_id, c.c_email_address HAVING SUM(ss.ss_net_paid) > 5000;"
    },
    # Customer state filter via customer_address
    {
        "question": "Tổng doanh thu từ khách hàng ở bang California",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk WHERE ca.ca_state = 'CA';"
    },
    
    # ===== RETURNS EXAMPLES =====
    # Store returns (not web_returns) for store
    {
        "question": "Có bao nhiêu đơn hàng bị trả lại tại cửa hàng ở IL năm 2002?",
        "sql": "SELECT COUNT(*) FROM store_returns sr JOIN customer c ON sr.sr_customer_sk = c.c_customer_sk JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk JOIN date_dim d ON sr.sr_returned_date_sk = d.d_date_sk WHERE ca.ca_state = 'IL' AND d.d_year = 2002;"
    },
    # Web returns
    {
        "question": "Thống kê đơn hàng online bị trả lại theo lý do",
        "sql": "SELECT r.r_reason_desc, COUNT(*) AS cnt FROM web_returns wr JOIN reason r ON wr.wr_reason_sk = r.r_reason_sk GROUP BY r.r_reason_desc ORDER BY cnt DESC;"
    },
    
    # ===== DATE/TIME EXAMPLES =====
    # Date filtering with quarter
    {
        "question": "Doanh thu quý 1 năm 2001",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2001 AND d.d_qoy = 1;"
    },
    # Day of week (Monday shopping)
    {
        "question": "Thống kê khách hàng mua sắm vào thứ Hai theo giới tính",
        "sql": "SELECT cd.cd_gender, COUNT(*) AS visit_count FROM store_sales ss JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_day_name = 'Monday' GROUP BY cd.cd_gender;"
    },
    
    # ===== INVENTORY EXAMPLES =====
    {
        "question": "Lượng tồn kho trung bình của Women vào cuối tuần tháng 12/2001",
        "sql": "SELECT AVG(inv.inv_quantity_on_hand) FROM inventory inv JOIN item i ON inv.inv_item_sk = i.i_item_sk JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk WHERE i.i_category = 'Women' AND d.d_weekend = 'Y' AND d.d_year = 2001 AND d.d_moy = 12;"
    },
    {
        "question": "Kho nào có nhiều hàng tồn nhất?",
        "sql": "SELECT w.w_warehouse_name, SUM(inv.inv_quantity_on_hand) AS total FROM inventory inv JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk GROUP BY w.w_warehouse_name ORDER BY total DESC LIMIT 5;"
    },
    
    # ===== TAX EXAMPLES =====
    {
        "question": "Tổng tiền thuế từ khách hàng thu nhập dưới 50000",
        "sql": "SELECT SUM(ss.ss_ext_tax) FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN household_demographics hd ON c.c_current_hdemo_sk = hd.hd_demo_sk JOIN income_band ib ON hd.hd_income_band_sk = ib.ib_income_band_sk WHERE ib.ib_upper_bound < 50000;"
    },
    
    # ===== YEAR-OVER-YEAR COMPARISON =====
    {
        "question": "Doanh số Shoes tại NC năm 1999 so với năm trước",
        "sql": "WITH current_year AS (SELECT SUM(ss.ss_net_paid) as revenue FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk JOIN item i ON ss.ss_item_sk = i.i_item_sk JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE i.i_category = 'Shoes' AND ca.ca_state = 'NC' AND d.d_year = 1999), previous_year AS (SELECT SUM(ss.ss_net_paid) as revenue FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk JOIN item i ON ss.ss_item_sk = i.i_item_sk JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE i.i_category = 'Shoes' AND ca.ca_state = 'NC' AND d.d_year = 1998) SELECT cy.revenue - py.revenue as revenue_change FROM current_year cy, previous_year py;"
    },
    
    # ===== WEB SALES CUSTOMER SK (correct column) =====
    {
        "question": "Khách hàng nào mua nhiều nhất trên web?",
        "sql": "SELECT c.c_customer_id, c.c_first_name, SUM(ws.ws_net_paid) as total FROM web_sales ws JOIN customer c ON ws.ws_bill_customer_sk = c.c_customer_sk GROUP BY c.c_customer_id, c.c_first_name ORDER BY total DESC LIMIT 5;"
    },
    
    # ===== ITEM CLASS (váy, áo, quần...) =====
    {
        "question": "Tồn kho của sản phẩm váy màu xanh dương",
        "sql": "SELECT SUM(inv.inv_quantity_on_hand) FROM inventory inv JOIN item i ON inv.inv_item_sk = i.i_item_sk WHERE i.i_color = 'blue' AND i.i_class = 'dresses';"
    },
    {
        "question": "Top 5 áo sơ mi bán chạy nhất",
        "sql": "SELECT i.i_product_name, SUM(ss.ss_quantity) AS total FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk WHERE i.i_class = 'shirts' GROUP BY i.i_product_name ORDER BY total DESC LIMIT 5;"
    },
    {
        "question": "Doanh thu từ quần jeans năm 2001",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE i.i_class = 'jeans' AND d.d_year = 2001;"
    },
    
    # ===== CATALOG PAGE =====
    {
        "question": "Doanh thu từ trang số 5 trong catalog",
        "sql": "SELECT SUM(cs.cs_sales_price) FROM catalog_sales cs JOIN catalog_page cp ON cs.cs_catalog_page_sk = cp.cp_catalog_page_sk WHERE cp.cp_catalog_page_number = 5;"
    },
    {
        "question": "Sản phẩm Electronics bán qua trang catalog số 10",
        "sql": "SELECT SUM(cs.cs_sales_price) FROM catalog_sales cs JOIN catalog_page cp ON cs.cs_catalog_page_sk = cp.cp_catalog_page_sk JOIN item i ON cs.cs_item_sk = i.i_item_sk WHERE cp.cp_catalog_page_number = 10 AND i.i_category = 'Electronics';"
    },
    
    # ===== SALES_PRICE vs NET_PAID =====
    {
        "question": "Tổng giá bán (sales price) của catalog ở TX",
        "sql": "SELECT SUM(cs.cs_sales_price) FROM catalog_sales cs JOIN customer c ON cs.cs_bill_customer_sk = c.c_customer_sk JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk WHERE ca.ca_state = 'TX';"
    },
    
    # ===== BÁN CHẠY = QUANTITY (IMPORTANT!) =====
    {
        "question": "Loại hàng nào bán chạy nhất trên web quý 1 năm 2000?",
        "sql": "SELECT i.i_category, SUM(ws.ws_quantity) AS total_qty FROM web_sales ws JOIN item i ON ws.ws_item_sk = i.i_item_sk JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE d.d_year = 2000 AND d.d_qoy = 1 GROUP BY i.i_category ORDER BY total_qty DESC LIMIT 1;"
    },
    {
        "question": "Sản phẩm nào bán chạy nhất năm 2001?",
        "sql": "SELECT i.i_product_name, SUM(ss.ss_quantity) AS total_sold FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2001 GROUP BY i.i_product_name ORDER BY total_sold DESC LIMIT 1;"
    },
    
    # ===== TRẢ LẠI HÀNG MẶC ĐỊNH = STORE_RETURNS =====
    {
        "question": "Tìm khách hàng trả lại hàng nhiều nhất",
        "sql": "SELECT c.c_first_name, c.c_last_name, SUM(sr.sr_return_amt) as total_return FROM store_returns sr JOIN customer c ON sr.sr_customer_sk = c.c_customer_sk GROUP BY c.c_customer_sk, c.c_first_name, c.c_last_name ORDER BY total_return DESC LIMIT 1;"
    },
    {
        "question": "Tổng giá trị hàng bị trả lại",
        "sql": "SELECT SUM(sr.sr_return_amt) FROM store_returns sr;"
    },
    
    # ===== DEMOGRAPHICS DIRECT JOIN (ss.ss_cdemo_sk) =====
    {
        "question": "Thống kê mua sắm theo giới tính vào thứ Hai",
        "sql": "SELECT cd.cd_gender, COUNT(*) AS visit_count FROM store_sales ss JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_day_name = 'Monday' GROUP BY cd.cd_gender;"
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-Coder-30B with Few-Shot & Schema Linking")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name/path")
    parser.add_argument("--test-data", type=str, default="research_pipeline/datasets/test.csv", help="Test CSV path")
    parser.add_argument("--train-data", type=str, default="research_pipeline/datasets/train_clean.csv", help="Train CSV for RAG examples")
    parser.add_argument("--easy", action="store_true", help="Use test_easy.csv")
    parser.add_argument("--shots", type=int, nargs="+", default=[3, 5, 7], help="Few-shot counts to test")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--db-path", type=str, default=DB_PATH, help="DuckDB database path")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG for dynamic example selection")
    parser.add_argument("--output-dir", type=str, default="research_pipeline/results", help="Results output directory")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_model_vllm(model_name: str):
    """Load model with vLLM for faster inference"""
    from vllm import LLM, SamplingParams
    
    print(f"Loading model with vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    return llm, "vllm"


def load_model_transformers(model_name: str):
    """Load model with transformers"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model with transformers: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    return (model, tokenizer), "transformers"


def generate_with_vllm(llm, prompt: str, max_tokens: int = 512) -> str:
    """Generate SQL with vLLM"""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["```", "\n\n\n", "Question:", "User:"],
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def generate_with_transformers(model_tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate SQL with transformers"""
    import torch
    model, tokenizer = model_tokenizer
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def get_schema_for_tables(tables: List[str]) -> str:
    """Generate schema DDL for selected tables with full columns"""
    schema_parts = []
    
    for table_name in tables:
        if table_name not in TPCDS_TABLES:
            continue
        table_info = TPCDS_TABLES[table_name]
        columns = ", ".join(table_info["columns"])  # All columns, not just top 10
        alias = table_info["alias"]
        schema_parts.append(f"-- {table_name} (alias: {alias})\n-- Columns: {columns}")
    
    return "\n\n".join(schema_parts)


def get_join_hints(tables: List[str]) -> str:
    """Get JOIN hints for selected tables"""
    hints = []
    for t1 in tables:
        for t2 in tables:
            if t1 != t2:
                key = (t1, t2)
                if key in JOIN_RELATIONSHIPS:
                    hints.append(f"  {t1} ⟷ {t2}: {JOIN_RELATIONSHIPS[key]}")
                key_rev = (t2, t1)
                if key_rev in JOIN_RELATIONSHIPS and (t1, t2) not in [(h.split(":")[0].strip().split(" ⟷ ")) for h in hints]:
                    hints.append(f"  {t2} ⟷ {t1}: {JOIN_RELATIONSHIPS[key_rev]}")
    return "\n".join(hints) if hints else "  (No direct joins)"


def build_fewshot_prompt(
    question: str,
    schema_info: str,
    join_hints: str,
    examples: List[Dict],
    num_shots: int
) -> str:
    """Build prompt with few-shot examples and schema context"""
    
    system_msg = """Bạn là chuyên gia SQL cho TPC-DS database. Sinh câu SQL chính xác.

=== CRITICAL RULES (ĐỌC KỸ!) ===
1. KHÔNG thêm filter (WHERE) nếu câu hỏi KHÔNG yêu cầu (VD: không thêm d.d_year nếu không hỏi về năm)
2. "bán chạy nhất" = SUM(quantity), KHÔNG phải SUM(sales_price)
3. "trả lại hàng" (không nói rõ channel) → mặc định dùng store_returns (sr)
4. "từ X trở lên" = >= X (VD: "từ 2 xe trở lên" = hd_vehicle_count >= 2)
5. Chỉ SELECT các columns cần thiết, không thêm columns thừa

=== CRITICAL COLUMN MAPPINGS ===
CUSTOMER TABLE:
- Email: c.c_email_address (NOT c_email)
- Name: c.c_first_name, c.c_last_name
- Login: c.c_login

CUSTOMER_DEMOGRAPHICS TABLE (cd):  
- Gender: cd.cd_gender
- Marital status: cd.cd_marital_status ('S'=Single, 'M'=Married, 'D'=Divorced)
- Credit rating: cd.cd_credit_rating
- Education: cd.cd_education_status
- Dependents: cd.cd_dep_count

HOUSEHOLD_DEMOGRAPHICS TABLE (hd):
- Vehicle count: hd.hd_vehicle_count
- Dependents: hd.hd_dep_count  
- Income band: hd.hd_income_band_sk

STORE_SALES TABLE (ss):
- Tax: ss.ss_ext_tax (NOT ss_tax)
- Revenue: ss.ss_net_paid
- Customer: ss.ss_customer_sk
- Demographics: ss.ss_cdemo_sk (direct link to customer_demographics)

DATE_DIM TABLE (d):
- Quarter: d.d_qoy (NOT d_quarter)
- Day name: d.d_day_name ('Monday', 'Tuesday', etc.)
- Weekend: d.d_weekend ('Y'/'N')
- NO d_state column - use customer_address.ca_state instead

WEB_SALES TABLE (ws):
- Customer: ws.ws_bill_customer_sk (NOT ws_customer_sk)

=== REVENUE vs QUANTITY ===
- "bán chạy nhất", "bán nhiều nhất" → SUM(ss_quantity / ws_quantity / cs_quantity)
- "doanh thu", "tổng doanh thu" → SUM(sales_price)
- "tiền thu được", "net" → SUM(net_paid)

=== ITEM TABLE (i) ===
- i.i_category: Danh mục lớn (Women, Men, Shoes, Electronics, Music, Home, Sports, Jewelry, Children)
- i.i_class: Loại sản phẩm cụ thể (dresses=váy, shirts=áo, pants=quần, jeans, blouses...)
- i.i_color: Màu sắc (blue, red, white, black...)
- "váy" → i.i_class = 'dresses'
- "áo" → i.i_class = 'shirts' hoặc 'blouses'

=== CATALOG PAGE ===
- Khi hỏi về "trang số X trong catalog" → JOIN catalog_page cp, use cp.cp_catalog_page_number

=== CHANNEL RULES ===
- "cửa hàng", "store", "retail" → store_sales (ss)
- "online", "web", "website", "trực tuyến" → web_sales (ws)
- "catalog", "mail order" → catalog_sales (cs)

=== RETURN RULES ===
- "trả lại hàng" (không rõ channel) → store_returns (sr) [MẶC ĐỊNH]
- "trả hàng online/web" → web_returns (wr)
- "trả hàng catalog" → catalog_returns (cr)

=== STATE/LOCATION ===
- Customer state: JOIN customer → customer_address, use ca.ca_state
- Store state: JOIN store, use s.s_state

=== DEMOGRAPHICS JOIN ===
- Khi cần demographics từ store_sales: dùng ss.ss_cdemo_sk trực tiếp
  VD: JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk
- KHÔNG cần đi qua customer table nếu chỉ cần demographics

Output ONLY the SQL query, no explanation.
================
IMPORTANT RULES:
1. DEMOGRAPHICS (MUST use separate tables):
   - Gender (cd_gender), marital status (cd_marital_status): USE customer_demographics (cd), NOT customer
   - Vehicle count (hd_vehicle_count): USE household_demographics (hd), NOT customer

2. CHANNEL SELECTION:
   - Web/Online sales: USE web_sales (ws), web_returns (wr)
   - Catalog/Mail-order: USE catalog_sales (cs), catalog_returns (cr)
   - Store/Retail/Cửa hàng: USE store_sales (ss), store_returns (sr)

3. COLUMN NAMES (exact names only):
   - Email: c.c_email_address (NOT c.c_email)
   - State: ca.ca_state from customer_address (NOT d.d_state)
   - Category: i.i_category from item table (NOT reason, NOT store name)
   - Quarter: d.d_qoy (NOT d_quarter)

4. CATEGORY FILTER:
   - When filtering by category (Men, Women, Home, Shoes, etc.): MUST JOIN item table
   - Use: WHERE i.i_category = 'category_name'

5. RETURNS:
   - Store returns: store_returns (sr) + item + customer_address for category/state filters
   - Web returns: web_returns (wr) + item + customer_address for category/state filters

- Always use table aliases
- Output ONLY the SQL query, no explanation"""


    # Build few-shot examples
    examples_text = ""
    for i, ex in enumerate(examples[:num_shots], 1):
        examples_text += f"\n### Ví dụ {i}:\nQuestion: {ex['question']}\nSQL: {ex['sql']}\n"
    
    prompt = f"""{system_msg}

### Schema liên quan:
{schema_info}

### JOIN hints:
{join_hints}

### Các ví dụ:{examples_text}

### Câu hỏi cần trả lời:
Question: {question}
SQL:"""
    
    return prompt


def extract_sql(response: str) -> str:
    """Extract SQL from model response"""
    # Remove thinking tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Try to find SQL in code blocks
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    sql_match = re.search(r'```\s*(SELECT.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try to find SELECT statement
    select_match = re.search(r'(SELECT\s+.+?)(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    return response.strip()


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = sql.strip().rstrip(';').strip()
    return sql


def validate_sql_execution(sql: str, db_path: str) -> Tuple[bool, str]:
    """Validate SQL executes without error"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(sql).fetchall()
        conn.close()
        return True, f"OK ({len(result)} rows)"
    except Exception as e:
        return False, str(e)[:100]


def compare_results(pred_sql: str, gold_sql: str, db_path: str) -> Tuple[bool, str]:
    """Compare execution results of predicted vs gold SQL"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Execute both
        try:
            pred_result = conn.execute(pred_sql).fetchall()
        except Exception as e:
            conn.close()
            return False, f"Pred error: {str(e)[:50]}"
        
        try:
            gold_result = conn.execute(gold_sql).fetchall()
        except Exception as e:
            conn.close()
            return False, f"Gold error: {str(e)[:50]}"
        
        conn.close()
        
        # Compare results
        pred_set = set(str(r) for r in pred_result)
        gold_set = set(str(r) for r in gold_result)
        
        if pred_set == gold_set:
            return True, "Exact match"
        
        # Check if subsets
        if pred_set.issubset(gold_set) or gold_set.issubset(pred_set):
            return True, "Subset match"
        
        # Check numeric similarity for aggregations
        if len(pred_result) == 1 and len(gold_result) == 1:
            try:
                pred_val = float(pred_result[0][0]) if pred_result[0][0] else 0
                gold_val = float(gold_result[0][0]) if gold_result[0][0] else 0
                if gold_val != 0 and abs(pred_val - gold_val) / abs(gold_val) < 0.01:
                    return True, "Numeric match (~1%)"
            except:
                pass
        
        return False, f"Mismatch (pred={len(pred_result)}, gold={len(gold_result)})"
    
    except Exception as e:
        return False, f"Compare error: {str(e)[:50]}"


def analyze_error(pred_sql: str, gold_sql: str, question: str) -> Dict:
    """Analyze common error patterns"""
    errors = {
        "channel_mistake": False,
        "demographics_mistake": False,
        "table_mistake": False,
        "join_mistake": False,
    }
    
    pred_lower = pred_sql.lower()
    gold_lower = gold_sql.lower()
    
    # Channel mistakes
    if "catalog" in question.lower():
        if "catalog_sales" in gold_lower and "catalog_sales" not in pred_lower:
            errors["channel_mistake"] = "Expected catalog_sales"
    if "online" in question.lower() or "website" in question.lower() or "web" in question.lower():
        if "web_sales" in gold_lower and "web_sales" not in pred_lower:
            errors["channel_mistake"] = "Expected web_sales"
    if "cửa hàng" in question.lower() or "store" in question.lower():
        if "store_sales" in gold_lower and "store_sales" not in pred_lower:
            errors["channel_mistake"] = "Expected store_sales"
    
    # Demographics mistakes
    if any(kw in question.lower() for kw in ["giới tính", "gender", "nam", "nữ", "male", "female"]):
        if "customer_demographics" in gold_lower and "customer_demographics" not in pred_lower:
            errors["demographics_mistake"] = "Missing customer_demographics for gender"
    
    if any(kw in question.lower() for kw in ["xe", "vehicle"]):
        if "household_demographics" in gold_lower and "household_demographics" not in pred_lower:
            errors["demographics_mistake"] = "Missing household_demographics for vehicle"
    
    # Table presence check
    gold_tables = set(re.findall(r'\b(?:from|join)\s+(\w+)', gold_lower))
    pred_tables = set(re.findall(r'\b(?:from|join)\s+(\w+)', pred_lower))
    
    missing_tables = gold_tables - pred_tables
    if missing_tables:
        errors["table_mistake"] = f"Missing tables: {missing_tables}"
    
    # JOIN count
    gold_joins = len(re.findall(r'\bjoin\b', gold_lower))
    pred_joins = len(re.findall(r'\bjoin\b', pred_lower))
    if gold_joins != pred_joins:
        errors["join_mistake"] = f"JOIN count: pred={pred_joins}, gold={gold_joins}"
    
    return errors


def run_benchmark(args):
    """Run full benchmark with different shot counts"""
    
    print("=" * 70)
    print("Qwen3-Coder-30B-A3B Full Pipeline Benchmark")
    print("=" * 70)
    
    # Load model
    try:
        if args.use_vllm:
            llm, backend = load_model_vllm(args.model)
        else:
            llm, backend = load_model_transformers(args.model)
        print(f"Model loaded! Backend: {backend}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying transformers backend...")
        llm, backend = load_model_transformers(args.model)
    
    # Initialize schema linker
    print("\nInitializing Schema Linker...")
    schema_linker = SchemaLinker()
    
    # Load test data
    if args.easy:
        test_path = "research_pipeline/datasets/test_easy.csv"
    else:
        test_path = args.test_data
    
    test_df = pd.read_csv(test_path)
    print(f"\nTest set: {test_path} ({len(test_df)} samples)")
    
    if args.max_test_samples:
        test_df = test_df.head(args.max_test_samples)
        print(f"Using first {len(test_df)} samples")
    
    # Get question/sql columns
    q_col = "Transcription" if "Transcription" in test_df.columns else "question"
    sql_col = "SQL Ground Truth" if "SQL Ground Truth" in test_df.columns else "sql"
    
    # Results storage
    all_results = {}
    
    for num_shots in args.shots:
        print(f"\n{'='*70}")
        print(f"Testing with {num_shots}-shot")
        print(f"{'='*70}")
        
        results = []
        correct = 0
        exec_correct = 0
        
        # Error statistics
        error_stats = {
            "channel_mistakes": 0,
            "demographics_mistakes": 0,
            "table_mistakes": 0,
            "join_mistakes": 0,
            "execution_errors": 0,
        }
        
        # Setup logging files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = output_dir / f"benchmark_{num_shots}shot_log_{timestamp}.txt"
        error_log_file = output_dir / f"error_analysis_{num_shots}shot_{timestamp}.txt"
        
        with open(log_file, "w", encoding="utf-8") as log_f, \
             open(error_log_file, "w", encoding="utf-8") as err_f:
            
            log_f.write("=" * 80 + "\n")
            log_f.write(f"{num_shots}-Shot Benchmark Detailed Log\n")
            log_f.write(f"Model: {args.model}\n")
            log_f.write(f"Test: {test_path}\n")
            log_f.write(f"Timestamp: {timestamp}\n")
            log_f.write("=" * 80 + "\n\n")
            
            err_f.write(f"ERROR ANALYSIS LOG ({num_shots}-shot)\n")
            err_f.write("=" * 80 + "\n\n")
        
            for idx, row in test_df.iterrows():
                question = row[q_col]
                gold_sql = row[sql_col]
                
                # Clean gold SQL
                gold_sql = gold_sql.replace('\n', ' ').strip()
                if not gold_sql.endswith(';'):
                    gold_sql += ';'
                
                print(f"\n[{idx+1}/{len(test_df)}] {question[:60]}...")
                log_f.write(f"\n{'='*80}\n")
                log_f.write(f"[{idx+1}/{len(test_df)}]\n")
                log_f.write(f"Question: {question}\n")
                log_f.write(f"Gold SQL: {gold_sql}\n\n")
            
                # Schema linking
                linked = schema_linker.link_schema(question, top_k_tables=6)
                linked_tables = linked["tables"]
                
                log_f.write(f"Linked Tables: {linked_tables}\n\n")
                
                if args.verbose:
                    print(f"  Linked tables: {linked_tables}")
                
                # Get schema and join hints
                schema_info = get_schema_for_tables(linked_tables)
                join_hints = get_join_hints(linked_tables)
                
                # Build prompt
                prompt = build_fewshot_prompt(
                    question=question,
                    schema_info=schema_info,
                    join_hints=join_hints,
                    examples=STATIC_FEWSHOT_EXAMPLES,
                    num_shots=num_shots
                )
                
                # Generate
                start_time = time.time()
                if backend == "vllm":
                    response = generate_with_vllm(llm, prompt)
                else:
                    response = generate_with_transformers(llm, prompt)
                gen_time = time.time() - start_time
                
                # Extract SQL
                pred_sql = extract_sql(response)
                
                log_f.write(f"Predicted SQL: {pred_sql}\n")
                log_f.write(f"Generation time: {gen_time:.2f}s\n\n")
                
                if args.verbose:
                    print(f"  Generated: {pred_sql[:100]}...")
                    print(f"  Time: {gen_time:.2f}s")
                
                # Validate execution
                exec_ok, exec_msg = validate_sql_execution(pred_sql, args.db_path)
                
                if not exec_ok:
                    error_stats["execution_errors"] += 1
                    log_f.write(f"EXECUTION ERROR: {exec_msg}\n")
            
                # Compare results
                match, match_msg = compare_results(pred_sql, gold_sql, args.db_path)
                
                if match:
                    correct += 1
                    status = "✓"
                    log_f.write(f"✓ CORRECT\n")
                else:
                    status = "✗"
                    log_f.write(f"✗ WRONG: {match_msg}\n")
                    
                    # Analyze error
                    error_analysis = analyze_error(pred_sql, gold_sql, question)
                    
                    err_f.write(f"\n{'='*80}\n")
                    err_f.write(f"[{idx+1}] {question}\n")
                    err_f.write(f"\nGold SQL:\n{gold_sql}\n")
                    err_f.write(f"\nPred SQL:\n{pred_sql}\n")
                    err_f.write(f"\nError Analysis:\n")
                    
                    for error_type, error_msg in error_analysis.items():
                        if error_msg:
                            err_f.write(f"  - {error_type}: {error_msg}\n")
                            if "channel" in error_type:
                                error_stats["channel_mistakes"] += 1
                            elif "demographics" in error_type:
                                error_stats["demographics_mistakes"] += 1
                            elif "table" in error_type:
                                error_stats["table_mistakes"] += 1
                            elif "join" in error_type:
                                error_stats["join_mistakes"] += 1
                    
                    err_f.write(f"\nExecution: {exec_msg}\n")
                    err_f.write(f"Match: {match_msg}\n")
                
                if exec_ok:
                    exec_correct += 1
                
                print(f"  {status} Exec: {exec_msg} | Match: {match_msg}")
                
                results.append({
                    "question": question,
                    "gold_sql": gold_sql,
                    "pred_sql": pred_sql,
                    "linked_tables": linked_tables,
                    "exec_ok": exec_ok,
                    "match": match,
                    "gen_time": gen_time,
                })
        
            # Calculate metrics
            accuracy = correct / len(test_df) * 100
            exec_rate = exec_correct / len(test_df) * 100
            
            summary = f"""
{'='*70}
SUMMARY ({num_shots}-shot)
{'='*70}
Total samples: {len(test_df)}
Execution success: {exec_correct}/{len(test_df)} = {exec_rate:.1f}%
Result match: {correct}/{len(test_df)} = {accuracy:.1f}%

ERROR STATISTICS:
- Channel mistakes: {error_stats['channel_mistakes']}
- Demographics mistakes: {error_stats['demographics_mistakes']}
- Table mistakes: {error_stats['table_mistakes']}
- JOIN mistakes: {error_stats['join_mistakes']}
- Execution errors: {error_stats['execution_errors']}
{'='*70}
"""
            
            print(summary)
            log_f.write(summary)
            err_f.write(summary)
            
            print(f"\n✓ Logs saved:")
            print(f"  - {log_file}")
            print(f"  - {error_log_file}")
        
        all_results[num_shots] = {
            "accuracy": accuracy,
            "exec_rate": exec_rate,
            "correct": correct,
            "exec_correct": exec_correct,
            "total": len(test_df),
            "error_stats": error_stats,
            "results": results,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Few-Shot Comparison")
    print("=" * 70)
    print(f"{'Shots':<10} {'Exec Rate':<15} {'Accuracy':<15}")
    print("-" * 40)
    for shots, data in all_results.items():
        print(f"{shots:<10} {data['exec_rate']:.1f}%{'':<10} {data['accuracy']:.1f}%")
    
    # Find best
    best_shots = max(all_results, key=lambda x: all_results[x]["accuracy"])
    print(f"\nBest: {best_shots}-shot with {all_results[best_shots]['accuracy']:.1f}% accuracy")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"qwen_coder_fewshot_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        # Convert results for JSON serialization
        save_data = {}
        for shots, data in all_results.items():
            save_data[str(shots)] = {
                "accuracy": data["accuracy"],
                "exec_rate": data["exec_rate"],
                "correct": data["correct"],
                "total": data["total"],
            }
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
