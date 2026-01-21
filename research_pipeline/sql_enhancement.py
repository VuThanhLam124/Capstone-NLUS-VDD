"""
SQL Enhancement Module cho Text-to-SQL

Bao gồm:
1. Dynamic Few-shot: Chọn examples phù hợp với câu hỏi
2. Post-processing SQL: Validate và chuẩn hóa SQL output
3. Self-Correction: Retry khi gặp lỗi syntax
4. Business Rules cải tiến
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import duckdb


# ============================================================
# 1. DYNAMIC FEW-SHOT
# ============================================================

# Các category câu hỏi để chọn few-shot phù hợp
QUERY_CATEGORIES = {
    "customer_info": {
        "keywords": ["khách hàng", "customer", "tên", "email", "địa chỉ", "họ", "danh xưng", "sinh", "tuổi"],
        "tables": ["customer", "customer_address", "customer_demographics"]
    },
    "sales_revenue": {
        "keywords": ["doanh thu", "bán", "revenue", "tổng tiền", "doanh số", "mua", "đơn hàng", "giao dịch"],
        "tables": ["store_sales", "web_sales", "catalog_sales"]
    },
    "product_item": {
        "keywords": ["sản phẩm", "mặt hàng", "item", "danh mục", "category", "giá", "thương hiệu", "brand"],
        "tables": ["item", "inventory"]
    },
    "returns": {
        "keywords": ["trả lại", "return", "hoàn", "tổn thất", "loss"],
        "tables": ["store_returns", "web_returns", "catalog_returns"]
    },
    "demographics": {
        "keywords": ["giới tính", "hôn nhân", "tín dụng", "thu nhập", "độc thân", "nam", "nữ", "gender"],
        "tables": ["customer_demographics", "household_demographics"]
    },
    "time_based": {
        "keywords": ["năm", "quý", "tháng", "ngày", "thứ", "tuần", "year", "quarter"],
        "tables": ["date_dim", "time_dim"]
    }
}

# Few-shot examples đặc biệt cho các pattern hay sai
DYNAMIC_EXAMPLES = {
    "customer_info": [
        {
            "question": "Tìm khách hàng có danh xưng là Tiến sĩ",
            "sql": "SELECT c_customer_id, c_first_name, c_last_name, c_email_address FROM customer WHERE c_salutation = 'Dr.';"
        },
        {
            "question": "Liệt kê khách hàng sinh ra ở Nhật Bản",
            "sql": "SELECT c_customer_id, c_first_name, c_last_name, c_email_address FROM customer WHERE c_birth_country = 'JAPAN';"
        },
        {
            "question": "Tìm các khách hàng tên bắt đầu bằng chữ M",
            "sql": "SELECT c_customer_id, c_first_name, c_last_name FROM customer WHERE c_first_name LIKE 'M%';"
        }
    ],
    "sales_revenue": [
        {
            "question": "Tổng doanh thu từ cửa hàng năm 2002",
            "sql": "SELECT SUM(ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002;"
        },
        {
            "question": "Doanh thu từ web trong quý 1",
            "sql": "SELECT SUM(ws_net_paid) FROM web_sales ws JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE d.d_qoy = 1;"
        }
    ],
    "returns": [
        {
            "question": "Tổng giá trị hàng bị trả lại tại cửa hàng",
            "sql": "SELECT SUM(sr_return_amt) FROM store_returns;"
        },
        {
            "question": "Số đơn trả hàng qua web năm 2001",
            "sql": "SELECT COUNT(*) FROM web_returns wr JOIN date_dim d ON wr.wr_returned_date_sk = d.d_date_sk WHERE d.d_year = 2001;"
        }
    ],
    "demographics": [
        {
            "question": "Thống kê khách hàng theo tình trạng hôn nhân",
            "sql": "SELECT cd_marital_status, COUNT(*) FROM customer_demographics GROUP BY cd_marital_status;"
        },
        {
            "question": "Tìm khách hàng có xếp hạng tín dụng thấp",
            "sql": "SELECT cd_demo_sk, cd_gender, cd_marital_status, cd_credit_rating FROM customer_demographics WHERE cd_credit_rating = 'Low Risk';"
        }
    ],
    "product_item": [
        {
            "question": "Top 5 sản phẩm có giá cao nhất trong danh mục Books",
            "sql": "SELECT i_item_desc, i_current_price FROM item WHERE i_category = 'Books' ORDER BY i_current_price DESC LIMIT 5;"
        },
        {
            "question": "Số lượng tồn kho của kho số 1",
            "sql": "SELECT inv_item_sk, inv_quantity_on_hand FROM inventory WHERE inv_warehouse_sk = 1;"
        }
    ]
}


def classify_question(question: str) -> List[str]:
    """Phân loại câu hỏi theo category để chọn few-shot phù hợp"""
    question_lower = question.lower()
    matches = []
    
    for category, config in QUERY_CATEGORIES.items():
        score = 0
        for keyword in config["keywords"]:
            if keyword in question_lower:
                score += 1
        if score > 0:
            matches.append((category, score))
    
    # Sắp xếp theo điểm và trả về top categories
    matches.sort(key=lambda x: -x[1])
    return [m[0] for m in matches[:2]]  # Top 2 categories


def get_dynamic_examples(question: str, max_examples: int = 3) -> List[Dict[str, str]]:
    """Lấy few-shot examples phù hợp với câu hỏi"""
    categories = classify_question(question)
    
    examples = []
    for cat in categories:
        if cat in DYNAMIC_EXAMPLES:
            examples.extend(DYNAMIC_EXAMPLES[cat])
            if len(examples) >= max_examples:
                break
    
    return examples[:max_examples]


# ============================================================
# 2. POST-PROCESSING SQL
# ============================================================

# Các cột không tồn tại trong TPC-DS
INVALID_COLUMNS = {
    "ca_email_address": "c_email_address",  # email chỉ có trong customer
    "ca_birth_country": "c_birth_country",
    "cd_preferred_cust_flag": "c_preferred_cust_flag",
    "cd_salutation": "c_salutation",
    "cr_customer_sk": None,  # Không tồn tại
    "sr_net_loss": "sr_return_amt",
}

# Bảng không tồn tại
INVALID_TABLES = {
    "category": None,  # Dùng i_category trong item
}


def post_process_sql(sql: str, conn: Optional[duckdb.DuckDBPyConnection] = None) -> Tuple[str, List[str]]:
    """
    Post-process SQL để sửa lỗi phổ biến
    Returns: (processed_sql, list of warnings/fixes applied)
    """
    fixes = []
    processed = sql
    
    # 1. Sửa các cột không tồn tại
    for invalid_col, replacement in INVALID_COLUMNS.items():
        if invalid_col in processed.lower():
            if replacement:
                # Thay thế
                processed = re.sub(
                    rf'\b{invalid_col}\b',
                    replacement,
                    processed,
                    flags=re.IGNORECASE
                )
                fixes.append(f"Đã thay {invalid_col} -> {replacement}")
            else:
                fixes.append(f"Cảnh báo: Cột {invalid_col} không tồn tại")
    
    # 2. Sửa bảng không tồn tại
    for invalid_table, replacement in INVALID_TABLES.items():
        pattern = rf'\bJOIN\s+{invalid_table}\b'
        if re.search(pattern, processed, re.IGNORECASE):
            fixes.append(f"Cảnh báo: Bảng {invalid_table} không tồn tại, dùng i_category trong item")
    
    # 3. Sửa SELECT TOP -> LIMIT (cho DuckDB)
    top_match = re.match(r'SELECT\s+TOP\s+(\d+)', processed, re.IGNORECASE)
    if top_match:
        n = top_match.group(1)
        processed = re.sub(
            r'SELECT\s+TOP\s+\d+',
            'SELECT',
            processed,
            flags=re.IGNORECASE
        )
        if 'LIMIT' not in processed.upper():
            processed = processed.rstrip(';') + f' LIMIT {n};'
        fixes.append(f"Đã sửa SELECT TOP {n} -> LIMIT {n}")
    
    # 4. Đảm bảo kết thúc bằng semicolon
    processed = processed.strip()
    if processed and not processed.endswith(';'):
        processed += ';'
    
    # 5. Validate syntax nếu có connection
    if conn:
        try:
            conn.execute(f"EXPLAIN {processed}")
        except Exception as e:
            fixes.append(f"Lỗi syntax: {str(e)[:100]}")
    
    return processed, fixes


def remove_unnecessary_joins(sql: str, question: str) -> str:
    """Loại bỏ JOIN không cần thiết dựa trên câu hỏi"""
    # Nếu câu hỏi chỉ về customer và không đề cập address
    if "address" not in question.lower() and "địa chỉ" not in question.lower():
        if "ca_" not in question and "đang sống" not in question.lower():
            # Có thể loại bỏ JOIN customer_address nếu không cần
            pass  # Implement nếu cần
    
    return sql


# ============================================================
# 3. SELF-CORRECTION (Retry 1 lần)
# ============================================================

CORRECTION_PROMPT_TEMPLATE = """
SQL bạn sinh ra bị lỗi: {error}

SQL lỗi:
{sql}

Hãy sửa lỗi và sinh lại SQL đúng. Chú ý:
- Kiểm tra tên bảng và cột có tồn tại không
- Đảm bảo các alias được khai báo đúng
- CHỈ trả về SQL, không giải thích

SQL đã sửa:
"""


def build_correction_prompt(original_sql: str, error: str, question: str, schema: str) -> str:
    """Tạo prompt để sửa SQL lỗi"""
    return f"""Bạn là chuyên gia SQL. Câu hỏi gốc: {question}

DATABASE SCHEMA:
{schema[:2000]}

{CORRECTION_PROMPT_TEMPLATE.format(error=error, sql=original_sql)}"""


def should_retry(error: str) -> bool:
    """Kiểm tra xem có nên retry không dựa trên loại lỗi"""
    retryable_errors = [
        "Binder Error",
        "Parser Error", 
        "Catalog Error",
        "not found",
        "does not exist",
        "syntax error"
    ]
    return any(e.lower() in error.lower() for e in retryable_errors)


# ============================================================
# 4. BUSINESS RULES CẢI TIẾN
# ============================================================

ENHANCED_BUSINESS_RULES = """
=== QUY TẮC SQL CƠ BẢN ===
1. KHÔNG thêm filter (WHERE) nếu câu hỏi KHÔNG yêu cầu
2. KHÔNG bịa cột/bảng không có trong schema
3. Luôn dùng alias cho bảng VÀ PHẢI KHAI BÁO TRONG FROM/JOIN
4. Output ONLY the SQL query, no explanation
5. Nếu dùng alias (vd: t cho time_dim), PHẢI có JOIN time_dim t TRƯỚC!

=== SELECT COLUMNS (QUAN TRỌNG!) ===
- Chỉ SELECT các cột THỰC SỰ CẦN THIẾT cho câu trả lời
- Ví dụ: "Tìm tên khách hàng" -> SELECT c_first_name, c_last_name (KHÔNG thêm cột khác)
- Ví dụ: "Danh sách email" -> SELECT c_email_address (KHÔNG thêm tên)

=== QUỐC TỊCH vs NƠI Ở (CRITICAL!) ===
- "đến từ/quốc tịch/sinh ra ở" -> c_birth_country (trong bảng customer)
- "đang sống/cư trú" -> ca_country (trong customer_address)
- KHÔNG dùng customer_address cho quốc tịch!

=== CỘT KHÔNG TỒN TẠI ===
Các cột sau KHÔNG TỒN TẠI, TUYỆT ĐỐI KHÔNG DÙNG:
- ca_email_address -> dùng c_email_address (trong customer)
- cd_preferred_cust_flag -> dùng c_preferred_cust_flag (trong customer)
- category table -> dùng i_category (trong item)

=== JOIN RULES ===
- CHỈ JOIN khi THẬT SỰ CẦN
- Query đơn giản về 1 bảng -> KHÔNG cần JOIN
- Ví dụ: "Tìm khách hàng tên M" -> SELECT... FROM customer WHERE... (KHÔNG JOIN)
"""


def get_enhanced_rules() -> str:
    """Trả về business rules cải tiến"""
    return ENHANCED_BUSINESS_RULES


# ============================================================
# MAIN ENHANCEMENT FUNCTION
# ============================================================

def enhance_sql_generation(
    question: str,
    generate_fn,
    schema: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    max_retries: int = 1
) -> Tuple[str, Dict]:
    """
    Hàm tổng hợp để cải thiện SQL generation
    
    Args:
        question: Câu hỏi người dùng
        generate_fn: Function để sinh SQL (nhận question, schema, examples)
        schema: Database schema
        conn: DuckDB connection để validate
        max_retries: Số lần retry khi lỗi (mặc định 1)
    
    Returns:
        (sql, metadata) với metadata chứa thông tin về quá trình xử lý
    """
    metadata = {
        "dynamic_examples": [],
        "post_process_fixes": [],
        "retries": 0,
        "original_sql": None,
        "final_sql": None
    }
    
    # 1. Lấy dynamic few-shot examples
    examples = get_dynamic_examples(question, max_examples=3)
    metadata["dynamic_examples"] = [ex["question"] for ex in examples]
    
    # 2. Sinh SQL lần đầu
    sql = generate_fn(question, schema, examples)
    metadata["original_sql"] = sql
    
    # 3. Post-process
    processed_sql, fixes = post_process_sql(sql, conn)
    metadata["post_process_fixes"] = fixes
    
    # 4. Validate và retry nếu cần
    if conn and max_retries > 0:
        try:
            conn.execute(processed_sql)
        except Exception as e:
            error = str(e)
            if should_retry(error):
                metadata["retries"] = 1
                # Build correction prompt và retry
                correction_prompt = build_correction_prompt(
                    processed_sql, error, question, schema
                )
                # Gọi lại generate_fn với prompt sửa lỗi
                corrected_sql = generate_fn(correction_prompt, "", [])
                processed_sql, new_fixes = post_process_sql(corrected_sql, conn)
                metadata["post_process_fixes"].extend(new_fixes)
    
    metadata["final_sql"] = processed_sql
    return processed_sql, metadata
