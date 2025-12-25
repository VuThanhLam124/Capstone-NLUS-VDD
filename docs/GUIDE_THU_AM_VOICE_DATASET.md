# HƯỚNG DẪN THU ÂM DATASET VOICE CHO TPC-H & TPC-DS

## 📋 TỔNG QUAN

**Mục tiêu**: Thu thập 300 mẫu voice thực tế cho mỗi người
**Thời lượng mỗi mẫu**: 10-15 giây
**Tổng thời gian thu âm**: ~50-75 phút (300 mẫu × 10-15s)
**Định dạng**: Giọng nói tiếng Việt tự nhiên đọc các câu truy vấn database

---

## 🎯 CÁC NGUYÊN TẮC QUAN TRỌNG

### 1. ✅ NÊN LÀM
- **Đọc tự nhiên** như đang nói chuyện bình thường
- **Có cảm xúc** (tò mò, quan tâm, vội vàng, lịch sự...)
- **Thêm từ đệm** tự nhiên: "ờ", "à", "ừm", "để xem", "kiểu như"
- **Thay đổi ngữ điệu** giữa các lần đọc
- **Đọc với tốc độ khác nhau** (nhanh, chậm, bình thường)
- **Thêm ngữ cảnh thực tế** (ví dụ phía dưới)

### 2. ❌ KHÔNG NÊN
- Đọc máy móc, không có cảm xúc
- Đọc quá nhanh hoặc quá chậm bất thường
- Đọc như đang đọc slides
- Phát âm sai chủ ý (trừ khi mô phỏng user không rõ yêu cầu)
- Thu âm ở nơi ồn

---

## 🗣️ CÁC KIỂU ĐỌC CÂU QUERY (QUAN TRỌNG)

### KIỂU 1: Lịch sự - Hỏi thông tin
**Ví dụ gốc**: "Tìm top 10 sản phẩm thuộc danh mục Electronics có giá cao hơn 100 đô la"

**Các cách đọc**:
1. "Cho mình hỏi, tìm top 10 sản phẩm thuộc danh mục Electronics có giá cao hơn 100 đô la nhé"
2. "Xin hỏi, ờ... cho mình xem top 10 sản phẩm Electronics, à, giá cao hơn 100 đô la"
3. "Bạn ơi cho hỏi, tìm giúp mình top 10 sản phẩm Electronics có giá trên 100 đô la được không?"
4. "Cho tôi biết top 10 sản phẩm Electronics nào có giá cao hơn 100 đô la"

### KIỂU 2: Vội vàng - Cần nhanh
**Ví dụ gốc**: "Tổng doanh thu bán hàng trong năm 2020 là bao nhiêu"

**Các cách đọc**:
1. "Này, tổng doanh thu năm 2020 bao nhiêu?" *(nhanh)*
2. "Ê, cho tôi xem nhanh tổng doanh thu bán hàng năm 2020" *(gấp)*
3. "Tổng doanh thu năm 2020 là bao nhiêu vậy?" *(tò mò)*

### KIỂU 3: Không chắc chắn - Đang tìm kiếm thông tin
**Ví dụ gốc**: "Liệt kê 5 mặt hàng bán chạy nhất tháng 12"

**Các cách đọc**:
1. "Ừm... cho tôi xem, ờ, 5 mặt hàng bán chạy nhất tháng 12 nhé"
2. "Để xem... liệt kê cho mình 5 mặt hàng, à, bán chạy nhất tháng 12"
3. "Kiểu như là... tìm giúp tôi 5 mặt hàng nào bán chạy nhất tháng 12"

### KIỂU 4: Chính thức - Báo cáo công việc
**Ví dụ gốc**: "Hiển thị 10 cửa hàng có doanh số cao nhất năm 2021"

**Các cách đọc**:
1. "Cho tôi biết 10 cửa hàng có doanh số cao nhất năm 2021"
2. "Tôi cần thống kê 10 cửa hàng doanh số cao nhất năm 2021"
3. "Hiển thị danh sách 10 cửa hàng có doanh số cao nhất trong năm 2021"

### KIỂU 5: Thông thường - Hỏi đồng nghiệp
**Ví dụ gốc**: "Đếm số lượng khách hàng ở bang California"

**Các cách đọc**:
1. "Ê, đếm giúp tôi số khách hàng ở California đi"
2. "Số lượng khách hàng ở bang California là bao nhiêu vậy?"
3. "Cho mình hỏi, ờ... có bao nhiêu khách hàng ở California?"

---

## 📝 QUY TRÌNH THU ÂM THỰC TẾ

### Bước 1: Chuẩn bị
```
✅ Môi trường yên tĩnh (tắt quạt, điều hòa nếu ồn)
✅ Micro chất lượng tốt (headphone hoặc USB mic)
✅ Nước uống (để giọng không khô)
✅ Danh sách câu query đã chuẩn bị
```

### Bước 2: Thiết lập phần mềm thu âm
**Khuyến nghị**: Audacity, Adobe Audition, hoặc đơn giản: Google Recorder, Voice Recorder

**Cài đặt**:
- Sample Rate: 16000 Hz (hoặc 44100 Hz rồi downsample sau)
- Format: WAV hoặc MP3
- Mono (1 channel)

### Bước 3: Thu âm từng mẫu

**Quy trình cho MỖI câu query**:

1. **Đọc thầm** câu query 1 lần để hiểu rõ
2. **Chọn 1 phong cách** từ 5 kiểu trên
3. **Bấm record** và đọc
4. **Dừng** sau khi đọc xong
5. **Kiểm tra** độ dài (10-15s)
   - Nếu quá ngắn (<8s): thêm từ đệm, nói chậm lại
   - Nếu quá dài (>17s): bỏ bớt từ thừa, nói nhanh hơn
6. **Lưu** với tên file rõ ràng: `query_001.wav`, `query_002.wav`...

### Bước 4: Tạo đa dạng

**Để đạt 300 mẫu từ ~100 câu query gốc**, bạn cần:

| Phương pháp | Số lượng |
|------------|----------|
| Đọc mỗi câu 3 lần với 3 phong cách khác nhau | 100 × 3 = 300 |
| Hoặc: 100 câu × 2 giọng × 1.5 biến thể | 300 |

**Cách tạo biến thể**:
- **Lần 1**: Đọc bình thường, lịch sự
- **Lần 2**: Đọc nhanh hơn, vội vàng
- **Lần 3**: Đọc có từ đệm, không chắc chắn

---

## 💡 VÍ DỤ THỰC TẾ CHO 1 CÂU QUERY

**Query gốc**: 
```
Tìm top 10 sản phẩm thuộc danh mục Electronics có giá cao hơn 100 đô la
```

### Biến thể 1 - Lịch sự (10s)
> "Cho mình hỏi, tìm giúp mình top 10 sản phẩm thuộc danh mục Electronics có giá cao hơn 100 đô la nhé"

**Cách đọc**: 
- Tốc độ: Bình thường
- Giọng điệu: Lịch sự, tò mò
- Nhấn mạnh: "top 10", "Electronics", "100 đô la"

### Biến thể 2 - Vội vàng (8s)
> "Này, tìm nhanh cho tôi top 10 sản phẩm Electronics giá trên 100 đô"

**Cách đọc**:
- Tốc độ: Nhanh
- Giọng điệu: Khẩn trương
- Nhấn mạnh: "nhanh", "top 10"

### Biến thể 3 - Không chắc (13s)
> "Ừm... cho tôi xem, ờ, tìm top 10 sản phẩm... à, danh mục Electronics, có giá cao hơn 100 đô la"

**Cách đọc**:
- Tốc độ: Chậm, do dự
- Giọng điệu: Suy tư, không chắc
- Thêm từ đệm: "ừm", "ờ", "à"

---

## 🎬 SCRIPT THU ÂM MẪU (10 CÂU ĐẦU)

### Query 1: Tìm sản phẩm Electronics
**Gốc**: Tìm top 10 sản phẩm thuộc danh mục Electronics có giá cao hơn 100 đô la

**3 biến thể thu âm**:
1. "Cho mình hỏi, tìm top 10 sản phẩm Electronics có giá cao hơn 100 đô la"
2. "Này, tìm nhanh top 10 sản phẩm Electronics giá trên 100 đô"
3. "Ừm... tìm giúp tôi, ờ, top 10 sản phẩm Electronics giá cao hơn 100 đô la"

---

### Query 2: Tổng doanh thu
**Gốc**: Tổng doanh thu bán hàng trong năm 2020 là bao nhiêu

**3 biến thể**:
1. "Xin hỏi, tổng doanh thu bán hàng năm 2020 là bao nhiêu vậy?"
2. "Ê, tổng doanh thu năm 2020 bao nhiêu?" *(nhanh)*
3. "Cho tôi biết, ừm... tổng doanh thu bán hàng trong năm 2020"

---

### Query 3: Mặt hàng bán chạy
**Gốc**: Liệt kê 5 mặt hàng bán chạy nhất tháng 12

**3 biến thể**:
1. "Bạn ơi, cho tôi xem 5 mặt hàng bán chạy nhất tháng 12"
2. "Liệt kê nhanh 5 mặt hàng bán chạy tháng 12"
3. "Để xem... liệt kê cho mình, ờ, 5 mặt hàng bán chạy nhất tháng 12"

---

### Query 4: Cửa hàng doanh số cao
**Gốc**: Hiển thị 10 cửa hàng có doanh số cao nhất năm 2021

**3 biến thể**:
1. "Cho tôi biết 10 cửa hàng có doanh số cao nhất năm 2021"
2. "Hiển thị 10 cửa hàng doanh số cao nhất năm 2021 đi"
3. "Tôi muốn xem, ừm... 10 cửa hàng có doanh số cao nhất năm 2021"

---

### Query 5: Đếm khách hàng
**Gốc**: Đếm số lượng khách hàng ở bang California

**3 biến thể**:
1. "Cho mình hỏi, có bao nhiêu khách hàng ở California?"
2. "Đếm số khách hàng ở California giúp tôi"
3. "Ờ... số lượng khách hàng ở bang California là bao nhiêu nhỉ?"

---

### Query 6: Chi tiêu nhiều nhất
**Gốc**: Khách hàng nào đã chi tiêu nhiều nhất trong năm 2020

**3 biến thể**:
1. "Xin hỏi, khách hàng nào chi tiêu nhiều nhất trong năm 2020?"
2. "Khách nào chi nhiều nhất năm 2020?" *(nhanh)*
3. "Cho tôi xem, ừm... khách hàng nào đã chi tiêu nhiều nhất năm 2020"

---

### Query 7: Giá trị trung bình
**Gốc**: Tính giá trị trung bình của các đơn hàng trong danh mục Books

**3 biến thể**:
1. "Cho tôi biết giá trị trung bình các đơn hàng danh mục Books"
2. "Tính nhanh giá trị trung bình đơn hàng Books"
3. "Để xem... tính giá trị trung bình, ờ, của các đơn hàng trong danh mục Books"

---

### Query 8: Hàng tồn kho
**Gốc**: Số lượng hàng tồn kho của sản phẩm thuộc nhóm Electronics là bao nhiêu

**3 biến thể**:
1. "Cho mình hỏi, số lượng hàng tồn kho Electronics là bao nhiêu?"
2. "Hàng tồn kho Electronics bao nhiêu?" *(ngắn gọn)*
3. "Ừm... số lượng hàng tồn kho, à, của sản phẩm Electronics là bao nhiêu nhỉ?"

---

### Query 9: Sản phẩm giá thấp
**Gốc**: Tìm các mặt hàng Books có giá thấp hơn 50 đô la

**3 biến thể**:
1. "Bạn ơi, tìm giúp mình các mặt hàng Books có giá thấp hơn 50 đô la"
2. "Tìm mặt hàng Books giá dưới 50 đô"
3. "Cho tôi xem... ờ... các mặt hàng Books giá thấp hơn 50 đô la"

---

### Query 10: Hàng trả lại
**Gốc**: Tổng số lượng hàng trả lại trong tháng 6 năm 2021

**3 biến thể**:
1. "Cho tôi biết tổng số hàng trả lại trong tháng 6 năm 2021"
2. "Tổng hàng trả lại tháng 6 năm 2021 là bao nhiêu?"
3. "Ừm... tổng số lượng hàng trả lại, à, trong tháng 6 năm 2021"

---

## 🔧 CÔNG CỤ HỖ TRỢ

### 1. Sử dụng script Python có sẵn

Script `generate_tpcds_voice_dataset.py` đã hỗ trợ:
- ✅ Tự động thêm từ đệm (ờ, à, ừm...)
- ✅ Tự động thêm ngữ cảnh (Cho mình hỏi, Bạn ơi...)
- ✅ Tạo biến thể tự động

**Chạy để xem ví dụ**:
```bash
cd research_pipeline
python generate_tpcds_voice_dataset.py --num_queries 100 --target_samples 300
```

### 2. Checklist thu âm

```
□ Kiểm tra micro hoạt động
□ Nơi yên tĩnh
□ Có nước uống
□ Đã đọc qua danh sách câu query
□ Biết cách tạo 5 kiểu đọc khác nhau
□ Đã thử thu âm thử 1-2 mẫu
□ Đã kiểm tra độ dài 10-15s
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### ❗ Tránh những sai lầm này:

1. **Đọc quá nhanh**: Dưới 8 giây
   - ➡️ Giải pháp: Thêm từ đệm, đọc chậm lại

2. **Đọc quá chậm**: Trên 17 giây
   - ➡️ Giải pháp: Bỏ từ thừa, tăng tốc

3. **Giọng đều đều không thay đổi**
   - ➡️ Giải pháp: Mỗi lần đọc dùng 1 phong cách khác nhau

4. **Phát âm sai chuyên ngành**
   - Electronics đọc: "e-lec-trô-nịch" ✅
   - Books đọc: "búc" ✅
   - Query đọc: "quy-ơ-ri" hoặc "cờ-ve-ri" ✅

5. **Không kiểm tra file sau khi thu**
   - ➡️ Nghe lại ngay để đảm bảo chất lượng

---

## 📊 KẾ HOẠCH THỰC HIỆN

### Phương án A: Thu âm 1 lần (Nhanh)
- **Thời gian**: 1 buổi (2-3 giờ)
- **Cách làm**: Thu 100 câu × 3 lần = 300 mẫu
- **Ưu điểm**: Giọng đồng nhất
- **Nhược điểm**: Mệt, dễ sai

### Phương án B: Chia nhỏ (Khuyến nghị)
- **Thời gian**: 3 ngày × 30 phút/ngày
- **Cách làm**: Mỗi ngày thu 100 mẫu
  - Ngày 1: 100 mẫu phong cách lịch sự
  - Ngày 2: 100 mẫu phong cách vội vàng
  - Ngày 3: 100 mẫu phong cách không chắc
- **Ưu điểm**: Không mệt, chất lượng cao
- **Nhược điểm**: Mất nhiều ngày

---

## ✅ CHECKLIST HOÀN THÀNH

Sau khi thu âm xong 300 mẫu:

```
□ Có đủ 300 file audio
□ Mỗi file từ 10-15 giây
□ Đã nghe lại ít nhất 10% số mẫu (30 mẫu)
□ Không có tiếng ồn nền
□ Giọng rõ ràng, dễ nghe
□ Có sự đa dạng về:
  □ Tốc độ đọc (nhanh, chậm, bình thường)
  □ Cảm xúc (lịch sự, vội, không chắc...)
  □ Từ đệm (có/không có)
□ Đã đặt tên file đúng quy tắc
□ Đã backup dữ liệu
```

---

## 🎓 KẾT LUẬN

**Yếu tố quan trọng nhất**: Tự nhiên và đa dạng!

Hệ thống AI sẽ học tốt hơn nếu dữ liệu giống cách người thật nói chuyện. Đừng cố gắng đọc "chuẩn" mà hãy đọc "tự nhiên".

**Good luck với việc thu âm! 🎤🎉**

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề kỹ thuật:
1. Kiểm tra file `generate_tpcds_voice_dataset.py`
2. Xem ví dụ output trong thư mục `data/`
3. Tham khảo metadata.csv để hiểu cấu trúc

**Công thức thành công**: Tự nhiên + Đa dạng + Kiên nhẫn = Dataset chất lượng cao! ✨
