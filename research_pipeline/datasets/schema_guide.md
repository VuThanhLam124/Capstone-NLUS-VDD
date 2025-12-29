# TPC-DS Schema Understanding Guide

Tài liệu này mô tả cấu trúc và ý nghĩa của các bảng trong TPC-DS Data Warehouse.
Dùng để giúp model hiểu rõ hơn về schema.

## Tổng quan
TPC-DS là mô hình Data Warehouse cho một công ty bán lẻ với 3 kênh bán hàng:
- **Store Sales (ss)**: Bán hàng tại cửa hàng
- **Web Sales (ws)**: Bán hàng qua website
- **Catalog Sales (cs)**: Bán hàng qua catalog

## Dimension Tables (Bảng Chiều)

### customer (Khách hàng)
| Cột quan trọng | Ý nghĩa |
|----------------|---------|
| c_customer_sk | Primary key |
| c_customer_id | Mã khách hàng (text) |
| c_first_name, c_last_name | Họ tên |
| c_birth_country | Quốc gia sinh |
| c_current_addr_sk | FK đến customer_address |

### item (Sản phẩm)
| Cột quan trọng | Ý nghĩa |
|----------------|---------|
| i_item_sk | Primary key |
| i_item_id | Mã sản phẩm (text) |
| i_item_desc | Mô tả sản phẩm |
| i_category | Danh mục (Electronics, Books, Men, Women...) |
| i_brand | Thương hiệu |
| i_current_price | Giá hiện tại |

### date_dim (Thời gian)
| Cột quan trọng | Ý nghĩa |
|----------------|---------|
| d_date_sk | Primary key |
| d_date | Ngày cụ thể |
| d_year | Năm (1999-2002) |
| d_moy | Tháng (1-12) |
| d_qoy | Quý (1-4) |
| d_day_name | Tên ngày (Monday...) |

### store (Cửa hàng)
| Cột quan trọng | Ý nghĩa |
|----------------|---------|
| s_store_sk | Primary key |
| s_store_name | Tên cửa hàng |
| s_state | Bang (VD: CA, NY, TX) |
| s_city | Thành phố |

## Fact Tables (Bảng Sự kiện)

### store_sales (Doanh số cửa hàng)
| Cột quan trọng | Ý nghĩa |
|----------------|---------|
| ss_sold_date_sk | FK đến date_dim |
| ss_item_sk | FK đến item |
| ss_customer_sk | FK đến customer |
| ss_store_sk | FK đến store |
| ss_quantity | Số lượng mua |
| ss_net_paid | Số tiền đã thanh toán |

### web_sales (Doanh số web)
Tương tự store_sales nhưng cho kênh online:
- ws_sold_date_sk, ws_item_sk, ws_bill_customer_sk
- ws_quantity, ws_net_paid

### catalog_sales (Doanh số catalog)
Cho kênh catalog:
- cs_sold_date_sk, cs_item_sk, cs_bill_customer_sk
- cs_quantity, cs_net_paid

## Return Tables (Bảng Trả hàng)
- **store_returns**: Trả hàng cửa hàng (sr_*)
- **web_returns**: Trả hàng web (wr_*)
- **catalog_returns**: Trả hàng catalog (cr_*)

## Quy tắc đặt tên cột
- Prefix = viết tắt tên bảng (ss_, ws_, cs_, sr_...)
- _sk = Surrogate Key (số nguyên, dùng làm FK)
- _id = Business ID (text)
- _name, _desc = Tên/mô tả
- _qty, _quantity = Số lượng
- _amt, _price, _paid = Số tiền

## Ví dụ JOIN phổ biến
```sql
-- Doanh thu theo năm
SELECT d.d_year, SUM(ss.ss_net_paid)
FROM store_sales ss
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
GROUP BY d.d_year;

-- Top sản phẩm theo danh mục
SELECT i.i_category, SUM(ss.ss_quantity)
FROM store_sales ss
JOIN item i ON ss.ss_item_sk = i.i_item_sk
GROUP BY i.i_category;
```
