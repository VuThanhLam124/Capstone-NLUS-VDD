#!/usr/bin/env python3
"""
Generate enhanced training data for TPC-DS Text-to-SQL
Focuses on:
1. Channel disambiguation (store/web/catalog)
2. Table-column ownership (cd_gender vs c_gender, etc.)
3. Common mistakes prevention

Usage:
    python generate_enhanced_data.py
"""

import json
from pathlib import Path

# ========== CHANNEL DISAMBIGUATION EXAMPLES ==========
CHANNEL_EXAMPLES = [
    # Web channel
    {
        "question": "Tính tổng doanh thu từ kênh online trong năm 2001",
        "sql": "SELECT SUM(ws.ws_net_paid) FROM web_sales ws JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE d.d_year = 2001;"
    },
    {
        "question": "Bao nhiêu đơn hàng được đặt qua website trong quý 2 năm 2002?",
        "sql": "SELECT COUNT(*) FROM web_sales ws JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002 AND d.d_qoy = 2;"
    },
    {
        "question": "Top 5 sản phẩm bán chạy nhất trên trang web theo số lượng",
        "sql": "SELECT i.i_product_name, SUM(ws.ws_quantity) AS total FROM web_sales ws JOIN item i ON ws.ws_item_sk = i.i_item_sk GROUP BY i.i_product_name ORDER BY total DESC LIMIT 5;"
    },
    {
        "question": "Thống kê đơn hàng online bị trả lại theo lý do",
        "sql": "SELECT r.r_reason_desc, COUNT(*) AS cnt FROM web_returns wr JOIN reason r ON wr.wr_reason_sk = r.r_reason_sk GROUP BY r.r_reason_desc ORDER BY cnt DESC;"
    },
    {
        "question": "URL nào trên website có nhiều giao dịch nhất?",
        "sql": "SELECT wp.wp_url, COUNT(*) AS cnt FROM web_sales ws JOIN web_page wp ON ws.ws_web_page_sk = wp.wp_web_page_sk GROUP BY wp.wp_url ORDER BY cnt DESC LIMIT 10;"
    },
    
    # Catalog channel
    {
        "question": "Tổng doanh thu từ catalog trong năm 2000",
        "sql": "SELECT SUM(cs.cs_net_paid) FROM catalog_sales cs JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk WHERE d.d_year = 2000;"
    },
    {
        "question": "Đơn hàng catalog nào có giá trị cao nhất?",
        "sql": "SELECT cs.cs_order_number, cs.cs_net_paid FROM catalog_sales cs ORDER BY cs.cs_net_paid DESC LIMIT 10;"
    },
    {
        "question": "Tìm các đơn catalog bị trả lại qua call center",
        "sql": "SELECT cc.cc_name, COUNT(*) AS returns FROM catalog_returns cr JOIN call_center cc ON cr.cr_call_center_sk = cc.cc_call_center_sk GROUP BY cc.cc_name;"
    },
    {
        "question": "Thống kê đơn hàng qua danh mục theo tháng năm 2001",
        "sql": "SELECT d.d_moy, COUNT(*) AS orders, SUM(cs.cs_net_paid) AS revenue FROM catalog_sales cs JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk WHERE d.d_year = 2001 GROUP BY d.d_moy ORDER BY d.d_moy;"
    },
    
    # Store channel
    {
        "question": "Doanh thu cửa hàng năm 2002 là bao nhiêu?",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002;"
    },
    {
        "question": "Cửa hàng nào có doanh số cao nhất?",
        "sql": "SELECT s.s_store_name, SUM(ss.ss_net_paid) AS revenue FROM store_sales ss JOIN store s ON ss.ss_store_sk = s.s_store_sk GROUP BY s.s_store_name ORDER BY revenue DESC LIMIT 5;"
    },
    {
        "question": "Sản phẩm nào bị trả lại nhiều nhất ở cửa hàng?",
        "sql": "SELECT i.i_product_name, COUNT(*) AS returns FROM store_returns sr JOIN item i ON sr.sr_item_sk = i.i_item_sk GROUP BY i.i_product_name ORDER BY returns DESC LIMIT 10;"
    },
    {
        "question": "Tổng số giao dịch tại các cửa hàng ở bang California",
        "sql": "SELECT COUNT(*) FROM store_sales ss JOIN store s ON ss.ss_store_sk = s.s_store_sk WHERE s.s_state = 'CA';"
    },
]

# ========== TABLE-COLUMN OWNERSHIP EXAMPLES ==========
OWNERSHIP_EXAMPLES = [
    # Gender in customer_demographics (NOT customer)
    {
        "question": "Thống kê số lượng khách hàng theo giới tính",
        "sql": "SELECT cd.cd_gender, COUNT(DISTINCT c.c_customer_sk) AS cnt FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk GROUP BY cd.cd_gender;"
    },
    {
        "question": "Khách hàng nam mua hàng nhiều hơn hay nữ?",
        "sql": "SELECT cd.cd_gender, SUM(ss.ss_net_paid) AS total FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk GROUP BY cd.cd_gender ORDER BY total DESC;"
    },
    {
        "question": "Doanh thu từ khách hàng nữ trên website năm 2001",
        "sql": "SELECT SUM(ws.ws_net_paid) FROM web_sales ws JOIN customer c ON ws.ws_bill_customer_sk = c.c_customer_sk JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk WHERE cd.cd_gender = 'F' AND d.d_year = 2001;"
    },
    
    # Marital status in customer_demographics
    {
        "question": "Khách hàng đã ly hôn có bao nhiêu người?",
        "sql": "SELECT COUNT(DISTINCT c.c_customer_sk) FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_marital_status = 'D';"
    },
    {
        "question": "So sánh doanh thu giữa khách hàng độc thân và đã kết hôn",
        "sql": "SELECT cd.cd_marital_status, SUM(ss.ss_net_paid) AS revenue FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_marital_status IN ('S', 'M') GROUP BY cd.cd_marital_status;"
    },
    {
        "question": "Danh sách khách hàng góa phụ mua hàng online",
        "sql": "SELECT DISTINCT c.c_first_name, c.c_last_name FROM web_sales ws JOIN customer c ON ws.ws_bill_customer_sk = c.c_customer_sk JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_marital_status = 'W';"
    },
    
    # Credit rating in customer_demographics
    {
        "question": "Khách hàng có xếp hạng tín dụng thấp nhất",
        "sql": "SELECT c.c_first_name, c.c_last_name, cd.cd_credit_rating FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_credit_rating = 'Low Risk' LIMIT 10;"
    },
    {
        "question": "Doanh thu từ khách hàng có tín dụng tốt",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_credit_rating = 'Good';"
    },
    
    # Vehicle count in household_demographics (NOT customer)
    {
        "question": "Khách hàng có từ 2 xe trở lên",
        "sql": "SELECT c.c_first_name, c.c_last_name, hd.hd_vehicle_count FROM customer c JOIN household_demographics hd ON c.c_current_hdemo_sk = hd.hd_demo_sk WHERE hd.hd_vehicle_count >= 2;"
    },
    {
        "question": "Trung bình số xe của khách hàng theo thu nhập",
        "sql": "SELECT ib.ib_lower_bound, ib.ib_upper_bound, AVG(hd.hd_vehicle_count) AS avg_vehicles FROM household_demographics hd JOIN income_band ib ON hd.hd_income_band_sk = ib.ib_income_band_sk GROUP BY ib.ib_lower_bound, ib.ib_upper_bound ORDER BY ib.ib_lower_bound;"
    },
    
    # Dependent count
    {
        "question": "Khách hàng có từ 3 người phụ thuộc trở lên",
        "sql": "SELECT c.c_first_name, c.c_last_name, cd.cd_dep_count FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk WHERE cd.cd_dep_count >= 3;"
    },
]

# ========== DATE HANDLING EXAMPLES ==========
DATE_EXAMPLES = [
    {
        "question": "Doanh thu quý 1 năm 2001",
        "sql": "SELECT SUM(ss.ss_net_paid) FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2001 AND d.d_qoy = 1;"
    },
    {
        "question": "Bán hàng vào ngày thứ Hai nhiều nhất ở đâu?",
        "sql": "SELECT s.s_store_name, COUNT(*) AS sales FROM store_sales ss JOIN store s ON ss.ss_store_sk = s.s_store_sk JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_day_name = 'Monday' GROUP BY s.s_store_name ORDER BY sales DESC LIMIT 5;"
    },
    {
        "question": "Doanh thu cuối tuần so với ngày thường",
        "sql": "SELECT d.d_weekend, SUM(ss.ss_net_paid) AS revenue FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk GROUP BY d.d_weekend;"
    },
    {
        "question": "Tháng nào có doanh thu cao nhất năm 2002?",
        "sql": "SELECT d.d_moy, SUM(ss.ss_net_paid) AS revenue FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk WHERE d.d_year = 2002 GROUP BY d.d_moy ORDER BY revenue DESC LIMIT 1;"
    },
]

# ========== INVENTORY & WAREHOUSE EXAMPLES ==========
INVENTORY_EXAMPLES = [
    {
        "question": "Tồn kho sản phẩm Music tại kho Kansas",
        "sql": "SELECT SUM(inv.inv_quantity_on_hand) FROM inventory inv JOIN item i ON inv.inv_item_sk = i.i_item_sk JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk WHERE i.i_category = 'Music' AND w.w_state = 'KS';"
    },
    {
        "question": "Kho nào có nhiều hàng tồn nhất?",
        "sql": "SELECT w.w_warehouse_name, SUM(inv.inv_quantity_on_hand) AS total FROM inventory inv JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk GROUP BY w.w_warehouse_name ORDER BY total DESC LIMIT 5;"
    },
    {
        "question": "Sản phẩm nào hết hàng?",
        "sql": "SELECT DISTINCT i.i_product_name FROM item i LEFT JOIN inventory inv ON i.i_item_sk = inv.inv_item_sk WHERE inv.inv_quantity_on_hand IS NULL OR inv.inv_quantity_on_hand = 0;"
    },
]


def format_training_sample(question: str, sql: str) -> dict:
    """Format as chat messages for training"""
    system_msg = """Bạn là một chuyên gia SQL. Tạo câu SQL chính xác cho câu hỏi dựa trên TPC-DS schema.

IMPORTANT RULES:
- Gender, marital status, credit rating: USE customer_demographics (cd), NOT customer
- Vehicle count: USE household_demographics (hd), NOT customer
- Web sales: USE web_sales (ws) for online/website transactions
- Catalog sales: USE catalog_sales (cs) for mail-order/catalog transactions
- Store sales: USE store_sales (ss) for retail store transactions
- Quarter: USE d_qoy (NOT d_quarter)
- Day name: USE d_day_name (NOT d_weekday)"""
    
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": sql}
        ]
    }


def main():
    output_file = Path("research_pipeline/datasets/train_enhanced.jsonl")
    
    all_examples = []
    all_examples.extend(CHANNEL_EXAMPLES)
    all_examples.extend(OWNERSHIP_EXAMPLES)
    all_examples.extend(DATE_EXAMPLES)
    all_examples.extend(INVENTORY_EXAMPLES)
    
    print(f"Total enhanced examples: {len(all_examples)}")
    
    # Write JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            sample = format_training_sample(ex["question"], ex["sql"])
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved to: {output_file}")
    
    # Merge with existing training data
    existing_file = Path("research_pipeline/datasets/train_schema_aware.jsonl")
    merged_file = Path("research_pipeline/datasets/train_merged_enhanced.jsonl")
    
    if existing_file.exists():
        print(f"\nMerging with {existing_file}...")
        
        # Read existing
        existing_samples = []
        with open(existing_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_samples.append(json.loads(line))
        
        # Read enhanced
        enhanced_samples = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    enhanced_samples.append(json.loads(line))
        
        # Merge (enhanced first for priority)
        merged = enhanced_samples + existing_samples
        
        # Write merged
        with open(merged_file, 'w', encoding='utf-8') as f:
            for sample in merged:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Merged: {len(enhanced_samples)} + {len(existing_samples)} = {len(merged)} samples")
        print(f"Saved to: {merged_file}")


if __name__ == "__main__":
    main()
