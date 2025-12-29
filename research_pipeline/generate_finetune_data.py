import argparse
import csv
import random
from pathlib import Path

# Config
CATEGORIES = ["Electronics", "Books", "Home", "Clothing", "Sports", "Music", "Toys", "Jewelry", "Men", "Women", "Children"]
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "NY", "TX", "WA"]
YEARS = [1999, 2000, 2001, 2002]
MONTHS = list(range(1, 13))
QUARTERS = [1, 2, 3, 4]
LIMITS = [5, 10, 20, 50, 100]
PRICES = [50, 100, 200, 500, 1000]
MARITAL_STATUS = ["M", "S", "D", "W"]  # Married, Single, Divorced, Widowed
GENDERS = ["M", "F"]
EDUCATION = ["Primary", "Secondary", "College", "2 yr Degree", "4 yr Degree", "Advanced Degree"]
REASONS = ["did not like", "damaged", "wrong item", "wrong size", "too big", "too small"]
SHIP_MODES = ["AIR", "GROUND", "RAIL", "SHIP", "MAIL"]

FIELD_GENERATORS = {
    "limit": lambda: random.choice(LIMITS),
    "category": lambda: random.choice(CATEGORIES),
    "price": lambda: random.choice(PRICES),
    "year": lambda: random.choice(YEARS),
    "month": lambda: random.choice(MONTHS),
    "quarter": lambda: random.choice(QUARTERS),
    "state": lambda: random.choice(STATES),
    "marital_status": lambda: random.choice(MARITAL_STATUS),
    "gender": lambda: random.choice(GENDERS),
    "education": lambda: random.choice(EDUCATION),
    "reason": lambda: random.choice(REASONS),
    "ship_mode": lambda: random.choice(SHIP_MODES),
}

PREFIXES = [
    "", "Cho tôi biết", "Hãy tìm", "Liệt kê", "Thống kê", "Tính toán", "Làm ơn cho biết", "Tìm giúp tôi", "Hỏi về", "Truy vấn"
]

TEMPLATES = [
    # --- SALES (Store, Web, Catalog) ---
    {
        "fields": ["limit", "category", "year"],
        "question": "Top {limit} sản phẩm danh mục {category} bán chạy nhất năm {year}",
        "sql": """SELECT i.i_item_desc, SUM(ss.ss_quantity) AS total_sold
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE i.i_category = '{category}' AND d.d_year = {year}
GROUP BY i.i_item_desc
ORDER BY total_sold DESC
LIMIT {limit};"""
    },
    {
        "fields": ["year", "state"],
        "question": "Tổng doanh thu cửa hàng tại bang {state} năm {year}",
        "sql": """SELECT SUM(ss.ss_net_paid) AS total_revenue
FROM store_sales ss
JOIN store s ON ss.ss_store_sk = s.s_store_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE s.s_state = '{state}' AND d.d_year = {year};"""
    },
    {
        "fields": ["year", "quarter"],
        "question": "Doanh thu web sales quý {quarter} năm {year}",
        "sql": """SELECT SUM(ws.ws_net_paid)
FROM web_sales ws
JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
WHERE d.d_year = {year} AND d.d_qoy = {quarter};"""
    },
    {
        "fields": ["limit", "category"],
        "question": "Top {limit} khách hàng mua nhiều {category} nhất qua catalog",
        "sql": """SELECT c.c_first_name, c.c_last_name, SUM(cs.cs_quantity) as total_qty
FROM catalog_sales cs
JOIN item i ON cs.cs_item_sk = i.i_item_sk
JOIN customer c ON cs.cs_bill_customer_sk = c.c_customer_sk
WHERE i.i_category = '{category}'
GROUP BY c.c_first_name, c.c_last_name
ORDER BY total_qty DESC
LIMIT {limit};"""
    },
    # --- RETURNS ---
    {
        "fields": ["reason", "year"],
        "question": "Số lượng hàng trả lại cửa hàng vì lý do '{reason}' trong năm {year}",
        "sql": """SELECT COUNT(*)
FROM store_returns sr
JOIN reason r ON sr.sr_reason_sk = r.r_reason_sk
JOIN date_dim d ON sr.sr_returned_date_sk = d.d_date_sk
WHERE r.r_reason_desc = '{reason}' AND d.d_year = {year};"""
    },
    {
        "fields": ["limit", "category"],
        "question": "Top {limit} mặt hàng {category} bị trả lại nhiều nhất trên web",
        "sql": """SELECT i.i_item_desc, COUNT(*) as return_count
FROM web_returns wr
JOIN item i ON wr.wr_item_sk = i.i_item_sk
WHERE i.i_category = '{category}'
GROUP BY i.i_item_desc
ORDER BY return_count DESC
LIMIT {limit};"""
    },
    # --- INVENTORY ---
    {
        "fields": ["category", "state"],
        "question": "Tổng tồn kho sản phẩm {category} tại các kho ở bang {state}",
        "sql": """SELECT SUM(inv.inv_quantity_on_hand)
FROM inventory inv
JOIN item i ON inv.inv_item_sk = i.i_item_sk
JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
WHERE i.i_category = '{category}' AND w.w_state = '{state}';"""
    },
    # --- CUSTOMER DEMOGRAPHICS ---
    {
        "fields": ["education", "year"],
        "question": "Tổng chi tiêu của khách hàng có trình độ {education} năm {year}",
        "sql": """SELECT SUM(ss.ss_net_paid)
FROM store_sales ss
JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE cd.cd_education_status = '{education}' AND d.d_year = {year};"""
    },
    {
        "fields": ["marital_status", "category"],
        "question": "Khách hàng tình trạng hôn nhân '{marital_status}' thích mua {category} không",
        "sql": """SELECT COUNT(*)
FROM store_sales ss
JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE cd.cd_marital_status = '{marital_status}' AND i.i_category = '{category}';"""
    },
    {
        "fields": ["gender", "state"],
        "question": "Số lượng khách hàng {gender} sống tại bang {state}",
        "sql": """SELECT COUNT(*)
FROM customer c
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
WHERE ca.ca_state = '{state}' AND cd.cd_gender = '{gender}';"""
    },
    # --- PROMOTION & SHIP MODE ---
    {
        "fields": ["ship_mode", "year"],
        "question": "Doanh thu web sales sử dụng phương thức vận chuyển {ship_mode} năm {year}",
        "sql": """SELECT SUM(ws.ws_net_paid)
FROM web_sales ws
JOIN ship_mode sm ON ws.ws_ship_mode_sk = sm.sm_ship_mode_sk
JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
WHERE sm.sm_type = '{ship_mode}' AND d.d_year = {year};"""
    },
    {
        "fields": ["category"],
        "question": "Tỷ lệ đơn hàng {category} có khuyến mãi",
        "sql": """SELECT CAST(SUM(CASE WHEN ss.ss_promo_sk IS NOT NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_category = '{category}';"""
    },
    # --- COMPLEX JOINS ---
    {
        "fields": ["year", "limit"],
        "question": "Top {limit} khách hàng có tổng giá trị trả hàng cao nhất năm {year}",
        "sql": """SELECT c.c_customer_id, c.c_first_name, c.c_last_name, SUM(sr.sr_return_amt) as total_return
FROM store_returns sr
JOIN customer c ON sr.sr_customer_sk = c.c_customer_sk
JOIN date_dim d ON sr.sr_returned_date_sk = d.d_date_sk
WHERE d.d_year = {year}
GROUP BY c.c_customer_id, c.c_first_name, c.c_last_name
ORDER BY total_return DESC
LIMIT {limit};"""
    },
    {
        "fields": ["state", "category"],
        "question": "Tìm các mặt hàng {category} bán chạy hơn mức trung bình ở bang {state}",
        "sql": """WITH avg_sales AS (
    SELECT AVG(ss_quantity) as avg_qty
    FROM store_sales
)
SELECT i.i_item_desc, SUM(ss.ss_quantity) as total_qty
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN store s ON ss.ss_store_sk = s.s_store_sk
WHERE i.i_category = '{category}' AND s.s_state = '{state}'
GROUP BY i.i_item_desc
HAVING SUM(ss.ss_quantity) > (SELECT avg_qty FROM avg_sales);"""
    },
    {
        "fields": ["year"],
        "question": "Khách hàng nào mua hàng ở cả 3 kênh store, web, catalog trong năm {year}",
        "sql": """SELECT c.c_customer_id
FROM customer c
WHERE EXISTS (SELECT 1 FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE ss.ss_customer_sk = c.c_customer_sk AND d.d_year = {year})
AND EXISTS (SELECT 1 FROM web_sales ws JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE ws.ws_bill_customer_sk = c.c_customer_sk AND d.d_year = {year})
AND EXISTS (SELECT 1 FROM catalog_sales cs JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk WHERE cs.cs_bill_customer_sk = c.c_customer_sk AND d.d_year = {year});"""
    }
]

# --- ADVANCED TEMPLATES (Window Functions, CTEs) ---
ADVANCED_TEMPLATES = [
    {
        "fields": ["year", "category"],
        "question": "Tính tổng doanh thu theo tháng và tỷ lệ đóng góp của từng tháng trong năm {year} cho danh mục {category}",
        "sql": """SELECT d.d_moy, 
       SUM(ss.ss_net_paid) as monthly_sales,
       SUM(ss.ss_net_paid) * 100.0 / SUM(SUM(ss.ss_net_paid)) OVER () as pct_contribution
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_year = {year} AND i.i_category = '{category}'
GROUP BY d.d_moy
ORDER BY d.d_moy;"""
    },
    {
        "fields": ["year", "limit"],
        "question": "Xếp hạng {limit} cửa hàng theo doanh thu năm {year}, hiển thị cả rank",
        "sql": """SELECT s.s_store_name, 
       SUM(ss.ss_net_paid) as revenue,
       RANK() OVER (ORDER BY SUM(ss.ss_net_paid) DESC) as store_rank
FROM store_sales ss
JOIN store s ON ss.ss_store_sk = s.s_store_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_year = {year}
GROUP BY s.s_store_name
LIMIT {limit};"""
    },
    {
        "fields": ["education", "year"],
        "question": "Tính doanh thu tích lũy (running total) theo tháng cho nhóm khách hàng trình độ {education} năm {year}",
        "sql": """SELECT d.d_moy,
       SUM(ss.ss_net_paid) as revenue,
       SUM(SUM(ss.ss_net_paid)) OVER (ORDER BY d.d_moy) as cumulative_revenue
FROM store_sales ss
JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE cd.cd_education_status = '{education}' AND d.d_year = {year}
GROUP BY d.d_moy
ORDER BY d.d_moy;"""
    },
    {
        "fields": ["state", "year"],
        "question": "So sánh doanh thu cửa hàng năm {year} ở bang {state} với doanh thu trung bình các bang khác",
        "sql": """WITH state_revenues AS (
    SELECT s.s_state, SUM(ss.ss_net_paid) as revenue
    FROM store_sales ss
    JOIN store s ON ss.ss_store_sk = s.s_store_sk
    JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE d.d_year = {year}
    GROUP BY s.s_state
)
SELECT revenue as state_revenue,
       (SELECT AVG(revenue) FROM state_revenues WHERE s_state != '{state}') as other_states_avg
FROM state_revenues
WHERE s_state = '{state}';"""
    },
    {
        "fields": ["category", "year"],
        "question": "Tìm các sản phẩm danh mục {category} có doanh thu cao hơn doanh thu trung bình của danh mục đó trong năm {year}",
        "sql": """WITH cat_avg AS (
    SELECT AVG(item_revenue) as avg_rev
    FROM (
        SELECT i.i_item_sk, SUM(ss.ss_net_paid) as item_revenue
        FROM store_sales ss
        JOIN item i ON ss.ss_item_sk = i.i_item_sk
        JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
        WHERE d.d_year = {year} AND i.i_category = '{category}'
        GROUP BY i.i_item_sk
    ) sub
)
SELECT i.i_item_desc, SUM(ss.ss_net_paid) as revenue
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_year = {year} AND i.i_category = '{category}'
GROUP BY i.i_item_desc
HAVING SUM(ss.ss_net_paid) > (SELECT avg_rev FROM cat_avg)
ORDER BY revenue DESC;"""
    },
    {
        "fields": ["year", "limit"],
        "question": "Top {limit} khách hàng tăng trưởng chi tiêu mạnh nhất (year-over-year growth) so với năm trước đó",
        "sql": """WITH cust_sales AS (
    SELECT c.c_customer_sk, c.c_first_name, c.c_last_name, d.d_year, SUM(ss.ss_net_paid) as annual_spend
    FROM store_sales ss
    JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
    JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE d.d_year IN ({year} - 1, {year})
    GROUP BY c.c_customer_sk, c.c_first_name, c.c_last_name, d.d_year
)
SELECT t2.c_first_name, t2.c_last_name, 
       (t2.annual_spend - t1.annual_spend) as growth
FROM cust_sales t1
JOIN cust_sales t2 ON t1.c_customer_sk = t2.c_customer_sk
WHERE t1.d_year = {year} - 1 AND t2.d_year = {year}
ORDER BY growth DESC
LIMIT {limit};"""
    }
]

def generate_data(count=2000, templates=TEMPLATES, seed=42, start_id=1):
    random.seed(seed)
    data = []
    seen = set()
    
    template_indices = list(range(len(templates)))
    
    while len(data) < count:
        t_idx = random.choice(template_indices)
        template = templates[t_idx]
        
        ctx = {k: FIELD_GENERATORS[k]() for k in template["fields"]}
        
        # Add prefix occasionally
        base_q = template["question"].format(**ctx)
        prefix = random.choice(PREFIXES)
        if prefix:
            question = f"{prefix} {base_q[0].lower()}{base_q[1:]}"
        else:
            question = base_q
            
        sql = template["sql"].format(**ctx)
        
        # Dedup
        key = (question, sql)
        if key in seen:
            continue
        seen.add(key)
        
        data.append({
            "ID": f"gen_{start_id + len(data)}",
            "Transcription": question,
            "SQL Ground Truth": sql
        })
        
    return data

def merge_csv(files, output_path):
    all_data = []
    
    # Read all files
    for fpath in files:
        path = Path(fpath)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_data.append(row)
    
    # Write merged
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if all_data:
        keys = all_data[0].keys()
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"Merged {len(all_data)} records to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic-count", type=int, default=2000)
    parser.add_argument("--adv-count", type=int, default=1000)
    parser.add_argument("--merge-target", default="research_pipeline/datasets/train_merged.csv")
    args = parser.parse_args()
    
    # Generate Basic
    basic_data = generate_data(args.basic_count, TEMPLATES, seed=42, start_id=1)
    basic_path = Path("research_pipeline/data/generated_finetune_basic.csv")
    basic_path.parent.mkdir(parents=True, exist_ok=True)
    with open(basic_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Transcription", "SQL Ground Truth"])
        writer.writeheader()
        writer.writerows(basic_data)
    print(f"Generated {len(basic_data)} basic samples.")

    # Generate Advanced
    adv_data = generate_data(args.adv_count, ADVANCED_TEMPLATES, seed=99, start_id=len(basic_data)+1)
    adv_path = Path("research_pipeline/data/generated_finetune_advanced.csv")
    with open(adv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Transcription", "SQL Ground Truth"])
        writer.writeheader()
        writer.writerows(adv_data)
    print(f"Generated {len(adv_data)} advanced samples.")
    
    # Merge with original train
    ORIGINAL_TRAIN = "research_pipeline/data/train.csv"
    merge_csv([ORIGINAL_TRAIN, basic_path, adv_path], args.merge_target)

if __name__ == "__main__":
    main()
