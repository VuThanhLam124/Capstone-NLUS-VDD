"""
Script để tạo SQL queries tương ứng với các câu mẫu giọng nói.
Sử dụng TPC-DS schema cho e-commerce dataset.
"""

import csv
import re

# Mapping categories
CATEGORIES = ["Books", "Electronics", "Home", "Children", "Sports", "Shoes", "Women", "Men", "Music", "Jewelry"]

# Mapping states
STATES = ["TX", "IL", "KY", "GA", "NC", "KS", "VA", "IA", "MO"]

# Mapping education levels
EDUCATION_LEVELS = {
    "Primary": "Primary",
    "Secondary": "Secondary",
    "College": "College",
    "2 yr Degree": "2 yr Degree",
    "4 yr Degree": "4 yr Degree",
    "Advanced Degree": "Advanced Degree"
}

# Mapping channels
CHANNELS = {
    "Store": "store_sales",
    "Web": "web_sales",
    "Catalog": "catalog_sales"
}

def generate_query(question):
    """Sinh SQL query từ câu hỏi tiếng Việt."""
    
    question_lower = question.lower()
    
    # 1. Pattern: "Ai mua đồ X nhiều nhất ở Y vậy?"
    # Tìm top khách hàng mua nhiều nhất
    match = re.search(r'ai mua đồ (\w+) nhiều nhất ở (\w+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        return f"""SELECT 
    c.c_customer_id,
    c.c_first_name || ' ' || c.c_last_name as customer_name,
    SUM(ss.ss_quantity) as total_quantity,
    SUM(ss.ss_net_paid) as total_spent
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_category = '{category}'
  AND ca.ca_state = '{state}'
GROUP BY c.c_customer_id, c.c_first_name, c.c_last_name
ORDER BY total_quantity DESC
LIMIT 1;"""

    # 2. Pattern: "Check giùm tôi tồn kho của X ở chi nhánh Y"
    match = re.search(r'tồn kho của (\w+) ở chi nhánh (\w+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        return f"""SELECT 
    w.w_warehouse_name,
    w.w_state,
    i.i_category,
    SUM(inv.inv_quantity_on_hand) as total_inventory
FROM inventory inv
JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
JOIN item i ON inv.inv_item_sk = i.i_item_sk
WHERE i.i_category = '{category}'
  AND w.w_state = '{state}'
GROUP BY w.w_warehouse_name, w.w_state, i.i_category
ORDER BY total_inventory DESC;"""

    # 3. Pattern: "Tổng doanh thu của X tại bang Y năm Z là bao nhiêu?"
    match = re.search(r'tổng doanh thu của (\w+) tại bang (\w+) năm (\d+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        year = match.group(3)
        return f"""SELECT 
    SUM(ss.ss_net_paid) as total_revenue
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE i.i_category = '{category}'
  AND ca.ca_state = '{state}'
  AND d.d_year = {year};"""

    # 4. Pattern: "Thống kê tổng số lượng X được mua bởi những khách hàng có trình độ Y"
    match = re.search(r'thống kê tổng số lượng (\w+) được mua bởi những khách hàng có trình độ (.+?)\.?$', question_lower)
    if match:
        category = match.group(1).capitalize()
        education = match.group(2).strip()
        # Match education level
        for edu_key, edu_val in EDUCATION_LEVELS.items():
            if edu_key.lower() in education.lower():
                education = edu_val
                break
        return f"""SELECT 
    cd.cd_education_status,
    i.i_category,
    SUM(ss.ss_quantity) as total_quantity,
    COUNT(*) as total_transactions
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_category = '{category}'
  AND cd.cd_education_status = '{education}'
GROUP BY cd.cd_education_status, i.i_category;"""

    # 5. Pattern: "Cửa hàng nào ở X đứng đầu về doanh số Y vào tháng Z?"
    match = re.search(r'cửa hàng nào ở (\w+) đứng đầu về doanh số (\w+) vào tháng (\d+)', question_lower)
    if match:
        state = match.group(1).upper()
        category = match.group(2).capitalize()
        month = match.group(3)
        return f"""SELECT 
    s.s_store_name,
    s.s_city,
    s.s_state,
    SUM(ss.ss_net_paid) as total_sales
FROM store_sales ss
JOIN store s ON ss.ss_store_sk = s.s_store_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE s.s_state = '{state}'
  AND i.i_category = '{category}'
  AND d.d_moy = {month}
GROUP BY s.s_store_name, s.s_city, s.s_state
ORDER BY total_sales DESC
LIMIT 1;"""

    # 6. Pattern: "Trong năm X, khách hàng ở Y mua nhiều Z không?"
    match = re.search(r'trong năm (\d+), khách hàng ở (\w+) mua nhiều (\w+)', question_lower)
    if match:
        year = match.group(1)
        state = match.group(2).upper()
        category = match.group(3).capitalize()
        return f"""SELECT 
    ca.ca_state,
    i.i_category,
    COUNT(*) as total_transactions,
    SUM(ss.ss_quantity) as total_quantity,
    SUM(ss.ss_net_paid) as total_revenue
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE ca.ca_state = '{state}'
  AND i.i_category = '{category}'
  AND d.d_year = {year}
GROUP BY ca.ca_state, i.i_category;"""

    # 7. Pattern: "Khách hàng có bằng X tại Y thường mua Z qua kênh nào?"
    match = re.search(r'khách hàng có bằng (.+?) tại (\w+) thường mua (\w+) qua kênh nào', question_lower)
    if match:
        education = match.group(1).strip()
        state = match.group(2).upper()
        category = match.group(3).capitalize()
        for edu_key, edu_val in EDUCATION_LEVELS.items():
            if edu_key.lower() in education.lower():
                education = edu_val
                break
        return f"""WITH store_channel AS (
    SELECT 'Store' as channel, SUM(ss.ss_net_paid) as total_sales
    FROM store_sales ss
    JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN item i ON ss.ss_item_sk = i.i_item_sk
    WHERE cd.cd_education_status = '{education}'
      AND ca.ca_state = '{state}'
      AND i.i_category = '{category}'
),
web_channel AS (
    SELECT 'Web' as channel, SUM(ws.ws_net_paid) as total_sales
    FROM web_sales ws
    JOIN customer c ON ws.ws_bill_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN item i ON ws.ws_item_sk = i.i_item_sk
    WHERE cd.cd_education_status = '{education}'
      AND ca.ca_state = '{state}'
      AND i.i_category = '{category}'
),
catalog_channel AS (
    SELECT 'Catalog' as channel, SUM(cs.cs_net_paid) as total_sales
    FROM catalog_sales cs
    JOIN customer c ON cs.cs_bill_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN item i ON cs.cs_item_sk = i.i_item_sk
    WHERE cd.cd_education_status = '{education}'
      AND ca.ca_state = '{state}'
      AND i.i_category = '{category}'
)
SELECT * FROM store_channel
UNION ALL SELECT * FROM web_channel
UNION ALL SELECT * FROM catalog_channel
ORDER BY total_sales DESC;"""

    # 8. Pattern: "Liệt kê X thương hiệu Y được ưa chuộng nhất bởi khách hàng ở Z"
    match = re.search(r'liệt kê (\d+) thương hiệu (\w+) được ưa chuộng nhất bởi khách hàng ở (\w+)', question_lower)
    if match:
        limit = match.group(1)
        category = match.group(2).capitalize()
        state = match.group(3).upper()
        return f"""SELECT 
    i.i_brand,
    i.i_category,
    COUNT(*) as purchase_count,
    SUM(ss.ss_quantity) as total_quantity,
    SUM(ss.ss_net_paid) as total_revenue
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_category = '{category}'
  AND ca.ca_state = '{state}'
GROUP BY i.i_brand, i.i_category
ORDER BY purchase_count DESC
LIMIT {limit};"""

    # 9. Pattern: "Năm ngoái dân ở X mua Y nhiều hay ít?"
    match = re.search(r'năm ngoái dân ở (\w+) mua (\w+) nhiều hay ít', question_lower)
    if match:
        state = match.group(1).upper()
        category = match.group(2).capitalize()
        return f"""SELECT 
    d.d_year,
    ca.ca_state,
    i.i_category,
    COUNT(*) as total_transactions,
    SUM(ss.ss_quantity) as total_quantity,
    SUM(ss.ss_net_paid) as total_revenue,
    AVG(ss.ss_net_paid) as avg_transaction_value
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE ca.ca_state = '{state}'
  AND i.i_category = '{category}'
  AND d.d_year = (SELECT MAX(d_year) - 1 FROM date_dim WHERE d_date_sk IN (SELECT ss_sold_date_sk FROM store_sales))
GROUP BY d.d_year, ca.ca_state, i.i_category;"""

    # 10. Pattern: "Có bao nhiêu đơn hàng X bị trả lại tại Y vào năm Z?"
    match = re.search(r'bao nhiêu đơn hàng (\w+) bị trả lại tại (\w+) vào năm (\d+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        year = match.group(3)
        return f"""SELECT 
    COUNT(*) as total_returns,
    SUM(sr.sr_return_quantity) as total_returned_quantity,
    SUM(sr.sr_return_amt) as total_return_amount
FROM store_returns sr
JOIN customer c ON sr.sr_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON sr.sr_item_sk = i.i_item_sk
JOIN date_dim d ON sr.sr_returned_date_sk = d.d_date_sk
WHERE i.i_category = '{category}'
  AND ca.ca_state = '{state}'
  AND d.d_year = {year};"""

    # 11. Pattern: "Tìm email của các khách hàng VIP ở X đã mua đồ Y trên Z đô"
    match = re.search(r'email của các khách hàng vip ở (\w+) đã mua đồ (\w+) trên (\d+) đô', question_lower)
    if match:
        state = match.group(1).upper()
        category = match.group(2).capitalize()
        amount = match.group(3)
        return f"""SELECT 
    c.c_customer_id,
    c.c_first_name || ' ' || c.c_last_name as customer_name,
    c.c_email_address,
    SUM(ss.ss_net_paid) as total_spent
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE ca.ca_state = '{state}'
  AND i.i_category = '{category}'
GROUP BY c.c_customer_id, c.c_first_name, c.c_last_name, c.c_email_address
HAVING SUM(ss.ss_net_paid) > {amount}
ORDER BY total_spent DESC;"""

    # 12. Pattern: "Tìm X sản phẩm Y có lợi nhuận thấp nhất trong năm Z"
    match = re.search(r'tìm (\d+) sản phẩm (\w+) có lợi nhuận thấp nhất trong năm (\d+)', question_lower)
    if match:
        limit = match.group(1)
        category = match.group(2).capitalize()
        year = match.group(3)
        return f"""SELECT 
    i.i_item_id,
    i.i_product_name,
    i.i_category,
    SUM(ss.ss_net_profit) as total_profit,
    SUM(ss.ss_net_paid) as total_revenue
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE i.i_category = '{category}'
  AND d.d_year = {year}
GROUP BY i.i_item_id, i.i_product_name, i.i_category
ORDER BY total_profit ASC
LIMIT {limit};"""

    # 13. Pattern: "Giá bán trung bình của mặt hàng X trong năm Y là bao nhiêu?"
    match = re.search(r'giá bán trung bình của mặt hàng (\w+) trong năm (\d+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        year = match.group(2)
        return f"""SELECT 
    i.i_category,
    AVG(ss.ss_sales_price) as avg_sales_price,
    MIN(ss.ss_sales_price) as min_sales_price,
    MAX(ss.ss_sales_price) as max_sales_price
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE i.i_category = '{category}'
  AND d.d_year = {year}
GROUP BY i.i_category;"""

    # 14. Pattern: "Cho tôi biết top X khách hàng chi tiêu nhiều nhất cho mặt hàng Y tại Z"
    match = re.search(r'top (\d+) khách hàng chi tiêu nhiều nhất cho mặt hàng (\w+) tại (\w+)', question_lower)
    if match:
        limit = match.group(1)
        category = match.group(2).capitalize()
        state = match.group(3).upper()
        return f"""SELECT 
    c.c_customer_id,
    c.c_first_name || ' ' || c.c_last_name as customer_name,
    c.c_email_address,
    SUM(ss.ss_net_paid) as total_spent,
    COUNT(*) as total_transactions
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_category = '{category}'
  AND ca.ca_state = '{state}'
GROUP BY c.c_customer_id, c.c_first_name, c.c_last_name, c.c_email_address
ORDER BY total_spent DESC
LIMIT {limit};"""

    # 15. Pattern: "So sánh lợi nhuận của X giữa kênh Y và Z tại W"
    match = re.search(r'so sánh lợi nhuận của (\w+) giữa kênh (\w+) và (\w+) tại (\w+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        channel1 = match.group(2).capitalize()
        channel2 = match.group(3).capitalize()
        state = match.group(4).upper()
        return f"""WITH store_profit AS (
    SELECT 'Store' as channel, SUM(ss.ss_net_profit) as total_profit
    FROM store_sales ss
    JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN item i ON ss.ss_item_sk = i.i_item_sk
    WHERE i.i_category = '{category}' AND ca.ca_state = '{state}'
),
web_profit AS (
    SELECT 'Web' as channel, SUM(ws.ws_net_profit) as total_profit
    FROM web_sales ws
    JOIN customer c ON ws.ws_bill_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN item i ON ws.ws_item_sk = i.i_item_sk
    WHERE i.i_category = '{category}' AND ca.ca_state = '{state}'
),
catalog_profit AS (
    SELECT 'Catalog' as channel, SUM(cs.cs_net_profit) as total_profit
    FROM catalog_sales cs
    JOIN customer c ON cs.cs_bill_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN item i ON cs.cs_item_sk = i.i_item_sk
    WHERE i.i_category = '{category}' AND ca.ca_state = '{state}'
)
SELECT * FROM store_profit WHERE channel IN ('{channel1}', '{channel2}')
UNION ALL SELECT * FROM web_profit WHERE channel IN ('{channel1}', '{channel2}')
UNION ALL SELECT * FROM catalog_profit WHERE channel IN ('{channel1}', '{channel2}')
ORDER BY channel;"""

    # 16. Pattern: "Doanh số của X tại Y năm Z so với năm trước đó tăng bao nhiêu?"
    match = re.search(r'doanh số của (\w+) tại (\w+) năm (\d+) so với năm trước đó tăng', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        year = int(match.group(3))
        prev_year = year - 1
        return f"""WITH current_year AS (
    SELECT SUM(ss.ss_net_paid) as revenue
    FROM store_sales ss
    JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN item i ON ss.ss_item_sk = i.i_item_sk
    JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE i.i_category = '{category}' AND ca.ca_state = '{state}' AND d.d_year = {year}
),
previous_year AS (
    SELECT SUM(ss.ss_net_paid) as revenue
    FROM store_sales ss
    JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN item i ON ss.ss_item_sk = i.i_item_sk
    JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE i.i_category = '{category}' AND ca.ca_state = '{state}' AND d.d_year = {prev_year}
)
SELECT 
    {year} as current_year,
    cy.revenue as current_revenue,
    {prev_year} as previous_year,
    py.revenue as previous_revenue,
    cy.revenue - py.revenue as revenue_change,
    ROUND((cy.revenue - py.revenue) / py.revenue * 100, 2) as growth_pct
FROM current_year cy, previous_year py;"""

    # 17. Pattern: "Năm X thì kênh Y mang về bao nhiêu tiền?"
    match = re.search(r'năm (\d+) thì kênh (\w+) mang về bao nhiêu tiền', question_lower)
    if match:
        year = match.group(1)
        channel = match.group(2).capitalize()
        if channel == "Store":
            return f"""SELECT 
    d.d_year,
    'Store' as channel,
    SUM(ss.ss_net_paid) as total_revenue,
    COUNT(*) as total_transactions
FROM store_sales ss
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_year = {year}
GROUP BY d.d_year;"""
        elif channel == "Web":
            return f"""SELECT 
    d.d_year,
    'Web' as channel,
    SUM(ws.ws_net_paid) as total_revenue,
    COUNT(*) as total_transactions
FROM web_sales ws
JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
WHERE d.d_year = {year}
GROUP BY d.d_year;"""
        else:  # Catalog
            return f"""SELECT 
    d.d_year,
    'Catalog' as channel,
    SUM(cs.cs_net_paid) as total_revenue,
    COUNT(*) as total_transactions
FROM catalog_sales cs
JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk
WHERE d.d_year = {year}
GROUP BY d.d_year;"""

    # 18. Pattern: "Năm X, doanh thu từ kênh Y cao hơn hay thấp hơn kênh Z?"
    match = re.search(r'năm (\d+), doanh thu từ kênh (\w+) cao hơn hay thấp hơn kênh (\w+)', question_lower)
    if match:
        year = match.group(1)
        channel1 = match.group(2).capitalize()
        channel2 = match.group(3).capitalize()
        return f"""WITH store_rev AS (
    SELECT 'Store' as channel, SUM(ss.ss_net_paid) as revenue
    FROM store_sales ss
    JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE d.d_year = {year}
),
web_rev AS (
    SELECT 'Web' as channel, SUM(ws.ws_net_paid) as revenue
    FROM web_sales ws
    JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    WHERE d.d_year = {year}
),
catalog_rev AS (
    SELECT 'Catalog' as channel, SUM(cs.cs_net_paid) as revenue
    FROM catalog_sales cs
    JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk
    WHERE d.d_year = {year}
)
SELECT * FROM store_rev
UNION ALL SELECT * FROM web_rev
UNION ALL SELECT * FROM catalog_rev
ORDER BY revenue DESC;"""

    # 19. Pattern: "Tỷ lệ trả hàng của X ở Y có cao hơn mức trung bình hệ thống không?"
    match = re.search(r'tỷ lệ trả hàng của (\w+) ở (\w+) có cao hơn mức trung bình', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        return f"""WITH category_state_returns AS (
    SELECT 
        COUNT(sr.sr_item_sk) * 1.0 / COUNT(ss.ss_item_sk) as return_rate
    FROM store_sales ss
    LEFT JOIN store_returns sr ON ss.ss_item_sk = sr.sr_item_sk AND ss.ss_ticket_number = sr.sr_ticket_number
    JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN item i ON ss.ss_item_sk = i.i_item_sk
    WHERE i.i_category = '{category}' AND ca.ca_state = '{state}'
),
system_average AS (
    SELECT 
        COUNT(sr.sr_item_sk) * 1.0 / COUNT(ss.ss_item_sk) as avg_return_rate
    FROM store_sales ss
    LEFT JOIN store_returns sr ON ss.ss_item_sk = sr.sr_item_sk AND ss.ss_ticket_number = sr.sr_ticket_number
)
SELECT 
    '{category}' as category,
    '{state}' as state,
    csr.return_rate as category_state_return_rate,
    sa.avg_return_rate as system_average_return_rate,
    CASE WHEN csr.return_rate > sa.avg_return_rate THEN 'Cao hơn' ELSE 'Thấp hơn hoặc bằng' END as comparison
FROM category_state_returns csr, system_average sa;"""

    # 20. Pattern: "Trong quý 4 năm X, nhóm khách hàng Y chi bao nhiêu tiền cho Z?"
    match = re.search(r'trong quý 4 năm (\d+), nhóm khách hàng (.+?) chi bao nhiêu tiền cho (\w+)', question_lower)
    if match:
        year = match.group(1)
        education = match.group(2).strip()
        category = match.group(3).capitalize()
        for edu_key, edu_val in EDUCATION_LEVELS.items():
            if edu_key.lower() in education.lower():
                education = edu_val
                break
        return f"""SELECT 
    cd.cd_education_status,
    i.i_category,
    SUM(ss.ss_net_paid) as total_spent,
    COUNT(*) as total_transactions
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE cd.cd_education_status = '{education}'
  AND i.i_category = '{category}'
  AND d.d_year = {year}
  AND d.d_qoy = 4
GROUP BY cd.cd_education_status, i.i_category;"""

    # 21. Pattern: "Hàng X trả lại ở Y có nhiều không, xem số liệu năm Z thử"
    match = re.search(r'hàng (\w+) trả lại ở (\w+) có nhiều không.*năm (\d+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        year = match.group(3)
        return f"""SELECT 
    i.i_category,
    ca.ca_state,
    d.d_year,
    COUNT(*) as total_returns,
    SUM(sr.sr_return_quantity) as total_returned_quantity,
    SUM(sr.sr_return_amt) as total_return_amount,
    AVG(sr.sr_return_amt) as avg_return_amount
FROM store_returns sr
JOIN customer c ON sr.sr_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON sr.sr_item_sk = i.i_item_sk
JOIN date_dim d ON sr.sr_returned_date_sk = d.d_date_sk
WHERE i.i_category = '{category}'
  AND ca.ca_state = '{state}'
  AND d.d_year = {year}
GROUP BY i.i_category, ca.ca_state, d.d_year;"""

    # 22. Pattern: "Cho tôi xem danh sách sản phẩm X bán chạy nhất ở Y"
    match = re.search(r'danh sách sản phẩm (\w+) bán chạy nhất ở (\w+)', question_lower)
    if match:
        category = match.group(1).capitalize()
        state = match.group(2).upper()
        return f"""SELECT 
    i.i_item_id,
    i.i_product_name,
    i.i_brand,
    i.i_category,
    SUM(ss.ss_quantity) as total_quantity_sold,
    SUM(ss.ss_net_paid) as total_revenue
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_category = '{category}'
  AND ca.ca_state = '{state}'
GROUP BY i.i_item_id, i.i_product_name, i.i_brand, i.i_category
ORDER BY total_quantity_sold DESC
LIMIT 10;"""

    # Default fallback
    return "-- Query không thể tự động sinh được cho câu hỏi này. Vui lòng viết query thủ công."


def main():
    # Đọc file CSV gốc
    input_file = 'voice_samples_300.csv'
    output_file = 'voice_samples_300_with_queries.csv'
    
    questions = []
    queries = []
    
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)  # Bỏ qua header
        
        for row in reader:
            if row:
                question = row[0].strip()
                if question:
                    questions.append(question)
                    query = generate_query(question)
                    queries.append(query)
    
    # Ghi file CSV mới với query
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Câu truy vấn giọng nói', 'SQL Query'])
        
        for question, query in zip(questions, queries):
            writer.writerow([question, query])
    
    print(f"Đã tạo file {output_file} với {len(queries)} queries.")
    print(f"\nVí dụ 5 queries đầu tiên:")
    for i in range(min(5, len(queries))):
        print(f"\n--- Câu {i+1}: {questions[i]}")
        print(queries[i][:500] + "..." if len(queries[i]) > 500 else queries[i])


if __name__ == "__main__":
    main()
