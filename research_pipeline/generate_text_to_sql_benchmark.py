import argparse
import json
import random
from pathlib import Path


CATEGORIES = [
    "Electronics",
    "Books",
    "Home",
    "Clothing",
    "Sports",
    "Music",
    "Toys",
    "Jewelry",
]
STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
]
YEARS = [2019, 2020, 2021, 2022, 2023]
MONTHS = list(range(1, 13))
QUARTERS = [1, 2, 3, 4]
LIMITS = [3, 5, 10, 20, 50]
PRICES = [10, 20, 50, 100, 200, 500, 1000]


FIELD_GENERATORS = {
    "limit": lambda: random.choice(LIMITS),
    "category": lambda: random.choice(CATEGORIES),
    "price": lambda: random.choice(PRICES),
    "year": lambda: random.choice(YEARS),
    "month": lambda: random.choice(MONTHS),
    "quarter": lambda: random.choice(QUARTERS),
    "state": lambda: random.choice(STATES),
}


TEMPLATES = [
    {
        "fields": ["limit", "category", "price"],
        "question": "Tìm top {limit} sản phẩm thuộc danh mục {category} có giá cao hơn {price} đô la",
        "sql": (
            "SELECT i.i_item_id, i.i_item_desc, i.i_current_price "
            "FROM item i "
            "WHERE i.i_category = '{category}' AND i.i_current_price > {price} "
            "ORDER BY i.i_current_price DESC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["limit", "category", "price"],
        "question": "Liệt kê {limit} sản phẩm danh mục {category} có giá thấp hơn {price} đô la",
        "sql": (
            "SELECT i.i_item_id, i.i_item_desc, i.i_current_price "
            "FROM item i "
            "WHERE i.i_category = '{category}' AND i.i_current_price < {price} "
            "ORDER BY i.i_current_price ASC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["year"],
        "question": "Tổng doanh thu bán hàng kênh cửa hàng trong năm {year} là bao nhiêu",
        "sql": (
            "SELECT SUM(ss.ss_net_paid) AS total_revenue "
            "FROM store_sales ss "
            "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year};"
        ),
    },
    {
        "fields": ["year", "quarter"],
        "question": "Tổng doanh thu kênh web trong quý {quarter} năm {year} là bao nhiêu",
        "sql": (
            "SELECT SUM(ws.ws_net_paid) AS total_revenue "
            "FROM web_sales ws "
            "JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} AND d.d_qoy = {quarter};"
        ),
    },
    {
        "fields": ["year", "month"],
        "question": "Tổng doanh thu kênh catalog trong tháng {month} năm {year} là bao nhiêu",
        "sql": (
            "SELECT SUM(cs.cs_net_paid) AS total_revenue "
            "FROM catalog_sales cs "
            "JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} AND d.d_moy = {month};"
        ),
    },
    {
        "fields": ["category", "year", "month"],
        "question": "Tổng số lượng bán ra của danh mục {category} trong tháng {month} năm {year}",
        "sql": (
            "SELECT SUM(ss.ss_quantity) AS total_quantity "
            "FROM store_sales ss "
            "JOIN item i ON ss.ss_item_sk = i.i_item_sk "
            "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
            "WHERE i.i_category = '{category}' AND d.d_year = {year} AND d.d_moy = {month};"
        ),
    },
    {
        "fields": ["limit", "year"],
        "question": "Top {limit} khách hàng chi tiêu nhiều nhất trong năm {year}",
        "sql": (
            "SELECT c.c_customer_sk, c.c_first_name, c.c_last_name, SUM(ss.ss_net_paid) AS total_spent "
            "FROM store_sales ss "
            "JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk "
            "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} "
            "GROUP BY c.c_customer_sk, c.c_first_name, c.c_last_name "
            "ORDER BY total_spent DESC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["limit"],
        "question": "Top {limit} bang có số lượng khách hàng cao nhất",
        "sql": (
            "SELECT ca.ca_state, COUNT(*) AS customer_count "
            "FROM customer c "
            "JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk "
            "GROUP BY ca.ca_state "
            "ORDER BY customer_count DESC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["state"],
        "question": "Đếm số lượng khách hàng ở bang {state}",
        "sql": (
            "SELECT COUNT(*) AS customer_count "
            "FROM customer_address ca "
            "WHERE ca.ca_state = '{state}';"
        ),
    },
    {
        "fields": ["limit", "year"],
        "question": "Top {limit} cửa hàng có doanh thu cao nhất trong năm {year}",
        "sql": (
            "SELECT s.s_store_name, SUM(ss.ss_net_paid) AS total_revenue "
            "FROM store_sales ss "
            "JOIN store s ON ss.ss_store_sk = s.s_store_sk "
            "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} "
            "GROUP BY s.s_store_name "
            "ORDER BY total_revenue DESC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["limit", "year"],
        "question": "Top {limit} sản phẩm bán chạy nhất trong năm {year}",
        "sql": (
            "SELECT i.i_item_desc, SUM(ss.ss_quantity) AS total_qty "
            "FROM store_sales ss "
            "JOIN item i ON ss.ss_item_sk = i.i_item_sk "
            "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} "
            "GROUP BY i.i_item_desc "
            "ORDER BY total_qty DESC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["price"],
        "question": "Danh mục nào có giá bán trung bình cao hơn {price} đô la",
        "sql": (
            "SELECT i.i_category, AVG(ss.ss_sales_price) AS avg_price "
            "FROM store_sales ss "
            "JOIN item i ON ss.ss_item_sk = i.i_item_sk "
            "GROUP BY i.i_category "
            "HAVING AVG(ss.ss_sales_price) > {price};"
        ),
    },
    {
        "fields": ["limit", "year"],
        "question": "Top {limit} website có doanh thu cao nhất trong năm {year}",
        "sql": (
            "SELECT w.web_name, SUM(ws.ws_net_paid) AS total_revenue "
            "FROM web_sales ws "
            "JOIN web_site w ON ws.ws_web_site_sk = w.web_site_sk "
            "JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} "
            "GROUP BY w.web_name "
            "ORDER BY total_revenue DESC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["limit", "year"],
        "question": "Top {limit} call center có doanh thu catalog cao nhất năm {year}",
        "sql": (
            "SELECT cc.cc_name, SUM(cs.cs_net_paid) AS total_revenue "
            "FROM catalog_sales cs "
            "JOIN call_center cc ON cs.cs_call_center_sk = cc.cc_call_center_sk "
            "JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} "
            "GROUP BY cc.cc_name "
            "ORDER BY total_revenue DESC "
            "LIMIT {limit};"
        ),
    },
    {
        "fields": ["category"],
        "question": "Tổng tồn kho của danh mục {category} là bao nhiêu",
        "sql": (
            "SELECT i.i_category, SUM(inv.inv_quantity_on_hand) AS total_inventory "
            "FROM inventory inv "
            "JOIN item i ON inv.inv_item_sk = i.i_item_sk "
            "WHERE i.i_category = '{category}' "
            "GROUP BY i.i_category;"
        ),
    },
    {
        "fields": ["year", "month"],
        "question": "Có bao nhiêu hóa đơn bán hàng trong tháng {month} năm {year}",
        "sql": (
            "SELECT COUNT(DISTINCT ss.ss_ticket_number) AS order_count "
            "FROM store_sales ss "
            "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
            "WHERE d.d_year = {year} AND d.d_moy = {month};"
        ),
    },
]


PREFIXES = [
    "",
    "Cho mình hỏi",
    "Giúp mình",
    "Bạn ơi",
    "Nhờ bạn",
]


def make_context(fields):
    return {field: FIELD_GENERATORS[field]() for field in fields}


def add_prefix(question):
    prefix = random.choice(PREFIXES)
    if not prefix:
        return question
    return f"{prefix} {question[0].lower()}{question[1:]}"


def generate_queries(target=200, seed=42):
    random.seed(seed)
    queries = []
    seen = set()

    while len(queries) < target:
        template = random.choice(TEMPLATES)
        ctx = make_context(template["fields"])
        question = add_prefix(template["question"].format(**ctx))
        sql = template["sql"].format(**ctx)
        key = (question, sql)
        if key in seen:
            continue
        seen.add(key)
        queries.append(
            {
                "id": f"q{len(queries) + 1}",
                "text": question,
                "sql": sql,
            }
        )
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate Vietnamese text-to-SQL benchmark for TPC-DS.")
    parser.add_argument("--output", type=str, default="research_pipeline/data/test_queries_vi_200_v2.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    queries = generate_queries(target=args.count, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2))
    print(f"Wrote {len(queries)} queries to {out_path}")


if __name__ == "__main__":
    main()
