# 📋 TPC-DS Text-to-SQL Prompt Rules

Tài liệu này ghi lại tất cả các rules được sử dụng trong system prompt để hướng dẫn LLM sinh SQL chính xác cho TPC-DS database.

---

## 🚨 CRITICAL RULES (Quy tắc quan trọng nhất)

| # | Rule | Giải thích | Ví dụ |
|---|------|------------|-------|
| 1 | **KHÔNG thêm filter không cần thiết** | Nếu câu hỏi không đề cập đến năm/tháng/quý, KHÔNG thêm `d.d_year` | ❌ `WHERE d.d_year = 2000` khi không hỏi |
| 2 | **"bán chạy" = quantity** | "bán chạy nhất" dùng SUM(quantity), không phải SUM(sales_price) | `SUM(ss.ss_quantity)` |
| 3 | **"trả lại hàng" mặc định = store** | Không rõ channel → dùng `store_returns` | `FROM store_returns sr` |
| 4 | **"từ X trở lên" = >= X** | Bao gồm cả X | `hd_vehicle_count >= 2` |
| 5 | **Chỉ SELECT columns cần thiết** | Không thêm columns thừa | Chỉ SELECT những gì được hỏi |

---

## 📊 Column Mappings

### Customer Table (`c`)
| Thuộc tính | Column đúng | ❌ Sai |
|------------|-------------|--------|
| Email | `c.c_email_address` | `c.c_email` |
| Tên | `c.c_first_name`, `c.c_last_name` | |
| Login | `c.c_login` | |

### Customer Demographics Table (`cd`)
| Thuộc tính | Column | Giá trị |
|------------|--------|---------|
| Giới tính | `cd.cd_gender` | 'M', 'F' |
| Tình trạng hôn nhân | `cd.cd_marital_status` | 'S'=Single, 'M'=Married, 'D'=Divorced |
| Xếp hạng tín dụng | `cd.cd_credit_rating` | 'Low Risk', 'Medium Risk', 'High Risk' |
| Học vấn | `cd.cd_education_status` | 'Advanced Degree', 'College', etc. |
| Số người phụ thuộc | `cd.cd_dep_count` | Integer |

### Household Demographics Table (`hd`)
| Thuộc tính | Column |
|------------|--------|
| Số xe | `hd.hd_vehicle_count` |
| Số người phụ thuộc | `hd.hd_dep_count` |
| Thu nhập | `hd.hd_income_band_sk` |

### Store Sales Table (`ss`)
| Thuộc tính | Column đúng | ❌ Sai |
|------------|-------------|--------|
| Thuế | `ss.ss_ext_tax` | `ss.ss_tax` |
| Doanh thu | `ss.ss_net_paid` | |
| Customer SK | `ss.ss_customer_sk` | |
| Demographics SK | `ss.ss_cdemo_sk` | ← Dùng trực tiếp, không cần qua customer |

### Date Dim Table (`d`)
| Thuộc tính | Column đúng | ❌ Sai |
|------------|-------------|--------|
| Quý | `d.d_qoy` | `d.d_quarter` |
| Tên ngày | `d.d_day_name` | 'Monday', 'Tuesday', etc. |
| Cuối tuần | `d.d_weekend` | 'Y'/'N' |
| Bang | ❌ KHÔNG CÓ | Dùng `ca.ca_state` |

### Web Sales Table (`ws`)
| Thuộc tính | Column đúng | ❌ Sai |
|------------|-------------|--------|
| Customer | `ws.ws_bill_customer_sk` | `ws.ws_customer_sk` |

---

## 💰 Revenue vs Quantity

| Từ khóa trong câu hỏi | Column sử dụng |
|-----------------------|----------------|
| "bán chạy nhất", "bán nhiều nhất" | `SUM(ss_quantity)` / `SUM(ws_quantity)` / `SUM(cs_quantity)` |
| "doanh thu", "tổng doanh thu" | `SUM(sales_price)` |
| "tiền thu được", "net" | `SUM(net_paid)` |

---

## 👗 Item Table (`i`)

### Category vs Class
| Level | Column | Ví dụ |
|-------|--------|-------|
| Danh mục lớn | `i.i_category` | 'Women', 'Men', 'Shoes', 'Electronics', 'Music', 'Home', 'Sports', 'Jewelry', 'Children' |
| Loại cụ thể | `i.i_class` | 'dresses', 'shirts', 'pants', 'jeans', 'blouses' |
| Màu sắc | `i.i_color` | 'blue', 'red', 'white', 'black' |

### Mapping tiếng Việt → i_class
| Tiếng Việt | i_class |
|------------|---------|
| váy | `'dresses'` |
| áo sơ mi | `'shirts'` |
| áo kiểu | `'blouses'` |
| quần | `'pants'` |
| quần jeans | `'jeans'` |

---

## 🛒 Channel Rules

| Từ khóa | Table | Alias |
|---------|-------|-------|
| "cửa hàng", "store", "retail" | `store_sales` | `ss` |
| "online", "web", "website", "trực tuyến" | `web_sales` | `ws` |
| "catalog", "mail order" | `catalog_sales` | `cs` |

---

## 🔄 Return Rules

| Từ khóa | Table | Alias |
|---------|-------|-------|
| "trả lại hàng" (không rõ channel) | `store_returns` | `sr` ← **MẶC ĐỊNH** |
| "trả hàng online", "trả hàng web" | `web_returns` | `wr` |
| "trả hàng catalog" | `catalog_returns` | `cr` |

---

## 📍 State/Location

| Loại | Cách lấy |
|------|----------|
| Bang của khách hàng | `JOIN customer → customer_address` → `ca.ca_state` |
| Bang của cửa hàng | `JOIN store` → `s.s_state` |
| Bang của kho | `JOIN warehouse` → `w.w_state` |

---

## 🔗 Demographics JOIN

### Cách 1: Trực tiếp từ store_sales (KHUYẾN NGHỊ)
```sql
SELECT cd.cd_gender, COUNT(*)
FROM store_sales ss
JOIN customer_demographics cd ON ss.ss_cdemo_sk = cd.cd_demo_sk
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_day_name = 'Monday'
GROUP BY cd.cd_gender;
```

### Cách 2: Qua customer table (chỉ khi cần thông tin customer)
```sql
SELECT cd.cd_gender, COUNT(*)
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
...
```

---

## 📖 Catalog Page

Khi câu hỏi đề cập "trang số X trong catalog":
```sql
SELECT SUM(cs.cs_sales_price)
FROM catalog_sales cs
JOIN catalog_page cp ON cs.cs_catalog_page_sk = cp.cp_catalog_page_sk
WHERE cp.cp_catalog_page_number = 5;
```

---

## 📝 Few-Shot Examples Categories

Các ví dụ few-shot được tổ chức theo nhóm:

1. **Channel Examples**: catalog_sales, web_sales, store_sales
2. **Demographics Examples**: gender, marital status, vehicle count, credit rating
3. **Item/Product Examples**: category filter, brand by state, i_class
4. **Customer + Address Examples**: email, state filter
5. **Returns Examples**: store_returns, web_returns
6. **Date/Time Examples**: quarter, day of week
7. **Inventory Examples**: tồn kho, warehouse
8. **Tax Examples**: ss_ext_tax
9. **Year-over-Year**: so sánh năm
10. **Catalog Page**: trang số trong catalog
11. **Sales Price vs Net Paid**: phân biệt doanh thu

---

## 📈 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-08 | Initial rules |
| 1.1 | 2026-01-08 | Added i_class rules for váy/áo/quần |
| 1.2 | 2026-01-08 | Added catalog_page rules |
| 1.3 | 2026-01-08 | Added critical rules: no hallucinated filters, quantity vs sales_price, default store_returns |
