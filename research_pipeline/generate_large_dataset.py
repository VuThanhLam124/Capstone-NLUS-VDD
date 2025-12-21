import json
import random
import os

OUTPUT_FILE = "/kaggle/working/data/test_queries_vi_1000.json"
# Local path for testing
LOCAL_OUTPUT_FILE = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/data/test_queries_vi_1000.json"

# Data Lists for Randomization
YEARS = [2019, 2020, 2021, 2022, 2023]
MONTHS = range(1, 13)
CATEGORIES = ["Electronics", "Books", "Home", "Clothing", "Sports", "Music", "Toys", "Jewelry"]
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY"]

# Filler Words / Disfluencies for Natural Speech Augmentation
FILLERS = ["ờ", "à", "ừm", "hmmm", "để xem", "kiểu như là", "thì", "ờ thì"]
EMOTIONAL_PREFIXES = ["Cho mình hỏi", "Bạn ơi cho hỏi", "Tính giúp mình", "Tra cứu giúp tôi", "Ê", "Này"]

# Templates: (Question Template, SQL Template)
TEMPLATES = [
    (
        "Tìm top {limit} sản phẩm thuộc danh mục {category} có giá cao hơn {price} đô la",
        "SELECT i_item_id, i_current_price FROM item WHERE i_current_price > {price} AND i_category = '{category}' ORDER BY i_current_price DESC LIMIT {limit};"
    ),
    (
        "Đếm số lượng khách hàng ở bang {state}",
        "SELECT count(*) FROM customer_address WHERE ca_state = '{state}';"
    ),
    (
        "Tổng doanh thu bán hàng trong năm {year} là bao nhiêu",
        "SELECT sum(ss_net_paid) FROM store_sales, date_dim WHERE ss_sold_date_sk = d_date_sk AND d_year = {year};"
    ),
    (
        "Liệt kê {limit} mặt hàng bán chạy nhất tháng {month}",
        "SELECT i_item_desc, COUNT(*) FROM store_sales, item, date_dim WHERE ss_item_sk = i_item_sk AND ss_sold_date_sk = d_date_sk AND d_moy = {month} GROUP BY i_item_desc ORDER BY 2 DESC LIMIT {limit};"
    ),
    (
        "Khách hàng nào đã chi tiêu nhiều nhất trong năm {year}",
        "SELECT c_first_name, c_last_name, SUM(ss_net_paid) FROM customer, store_sales, date_dim WHERE c_customer_sk = ss_customer_sk AND ss_sold_date_sk = d_date_sk AND d_year = {year} GROUP BY c_first_name, c_last_name ORDER BY 3 DESC LIMIT 1;"
    ),
    (
        "Tìm danh sách các khách hàng sống tại {state} mua hàng trong năm {year}",
        "SELECT distinct c_first_name, c_last_name FROM customer, customer_address, store_sales, date_dim WHERE c_current_addr_sk = ca_address_sk AND ca_state = '{state}' AND c_customer_sk = ss_customer_sk AND ss_sold_date_sk = d_date_sk AND d_year = {year};"
    ),
    (
        "Tính giá trị trung bình của các đơn hàng trong danh mục {category}",
        "SELECT AVG(ss_net_paid) FROM store_sales, item WHERE ss_item_sk = i_item_sk AND i_category = '{category}';"
    ),
    (
        "Hiển thị {limit} cửa hàng có doanh số cao nhất năm {year}",
        "SELECT s_store_name, SUM(ss_net_paid) FROM store, store_sales, date_dim WHERE s_store_sk = ss_store_sk AND ss_sold_date_sk = d_date_sk AND d_year = {year} GROUP BY s_store_name ORDER BY 2 DESC LIMIT {limit};"
    ),
    (
        "Số lượng hàng tồn kho của sản phẩm thuộc nhóm {category} là bao nhiêu",
        "SELECT SUM(inv_quantity_on_hand) FROM inventory, item WHERE inv_item_sk = i_item_sk AND i_category = '{category}';"
    ),
    (
        "Tìm các mặt hàng {category} có giá thấp hơn {price} đô la",
        "SELECT i_item_desc FROM item WHERE i_category = '{category}' AND i_current_price < {price};"
    )
]

def inject_disfluencies(text):
    """
    Injects fillers and emotional prefixes to make text sound more natural/conversational.
    Strategy:
    1. 30% chance to add a prefix.
    2. 40% chance to inject a filler word at a comma or random space.
    """
    if random.random() < 0.3:
        text = f"{random.choice(EMOTIONAL_PREFIXES)} {text.lower()}"
    
    words = text.split()
    if len(words) > 4 and random.random() < 0.4:
        # Insert filler at random position (not start/end)
        idx = random.randint(1, len(words) - 2)
        words.insert(idx, f"... {random.choice(FILLERS)} ...")
        text = " ".join(words)
        
    return text.replace("... ...", "...") # Cleanup

def generate_dataset(num_samples=1000):
    data = []
    
    for i in range(1, num_samples + 1):
        # Pick a random template
        tmpl_q, tmpl_s = random.choice(TEMPLATES)
        
        # Generate Random Values
        ctx = {
            "limit": random.choice([3, 5, 10, 20, 50]),
            "category": random.choice(CATEGORIES),
            "price": random.choice([10, 50, 100, 200, 500, 1000]),
            "state": random.choice(STATES),
            "year": random.choice(YEARS),
            "month": random.choice(MONTHS)
        }
        
        # Fill Template
        question = tmpl_q.format(**ctx)
        sql = tmpl_s.format(**ctx)
        
        # Inject Natural Speech Augmentations
        question_natural = inject_disfluencies(question)
        
        data.append({
            "id": f"q{i}_vi",
            "text": question_natural,
            "original_text": question, # Keep original for reference
            "sql": sql
        })
        
    return data

if __name__ == "__main__":
    dataset = generate_dataset(1000)
    
    # Determine output path (Kaggle vs Local)
    out_path = OUTPUT_FILE if os.path.exists("/kaggle/working") else LOCAL_OUTPUT_FILE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Generated {len(dataset)} conversational queries to {out_path}")
