Quá trình khảo sát và thực nghiệm
Dataset và Methodology
Dataset Speech-to-Text:
Quy mô: 800 samples


Cấu trúc: ID, audio file (.mp3), transcription, SQL ground truth


Đặc điểm: Bao gồm cả giọng địa phương, tiếng lóng và thuật ngữ kỹ thuật


Dataset Text-to-SQL:
Loại Easy (100 câu): 46% unary operation, 54% với 1 JOIN, không có subquery/CTE/UNION


Loại Hard (90 câu): Trung bình 3.4 JOIN/câu (max 12), 20% subquery, 59% GROUP BY, 33% queries phức tạp (complexity ≥10)


Metrics đánh giá:
STT: Word Error Rate (WER) - tỷ lệ lỗi từ


Text-to-SQL: Tỷ lệ SQL syntax error và tỷ lệ output chính xác


Kết quả thực nghiệm Speech-to-Text
Mô hình
WER
API Gemini Flash 3 Pro
0.05
Whisper-large-v3 (OpenAI)
0.06
Chunkformer (KhanhLD)
0.09
PhoWhisper-large (VinAI)
0.10

Phân tích lỗi phổ biến:
Âm thanh ngập ngừng (ờ, ừ, ừm) bị bỏ qua


Lỗi phát âm theo vùng miền (giọng địa phương)


Chuyển đổi chữ-số không nhất quán (chín vs. 9)


Việt hóa từ tiếng Anh (select → "sơ lếch" với PhoWhisper)


Kết quả thực nghiệm Text-to-SQL
Model
Cấu hình
Accuracy
Qwen-3-4b-GGUF
Chưa fine-tune + few-shot 3 + schema linking
~17%
Qwen-3-4b-GGUF
Đã fine-tune + few-shot 3 + schema linking
~33%
Qwen3-Coder-30B
Few-shot 3 + schema linking
~63.4%
Qwen3-Coder-30B
Few-shot 5 + schema linking
~54%
DeepSeek-Coder-V2
Few-shot 5 + schema linking
~56%

Nhận xét quan trọng:
Fine-tuning cải thiện gần gấp đôi accuracy cho model nhỏ (17% → 33%)


Model lớn hơn (30B) cho kết quả tốt hơn đáng kể ngay cả khi chưa fine-tune


Few-shot 3 tối ưu hơn few-shot 5 với model lớn, tránh over-constraint


Đang tiến hành fine-tuning cho Qwen3-Coder-30B và DeepSeek-Coder-V2 để cải thiện thêm


Phân tích lỗi:
SQL syntax error: Thiếu dấu ngoặc, sai keyword


Bịa tên bảng/cột: Model hallucinate khi không tìm thấy trong schema


Kết quả logic sai: JOIN không đúng, WHERE condition sai


Chi phí và Infrastructure
Training:
GPU: RTX A5880 Ada 48GB VRAM, Disk 150GB


Chi phí: 0.585$/giờ (Vast.ai)


Inference (API Gemini):
Audio: 0.3-1$/triệu token (~8.5 giờ audio)


Text output: 0.4$/triệu token (~2.5-3.2 triệu ký tự)


Quản trị rủi ro và Chiến lược triển khai
Unhappy Cases và Giải pháp
Speech-to-Text:
Nhận diện sai → Thêm UI verify/edit cho người dùng


Truyền đạt không mạch lạc → Hỗ trợ rewrite/paraphrase bằng LLM


Bất đồng ngôn ngữ → Sử dụng Whisper đa ngôn ngữ hoặc API Gemini


Giọng địa phương → Fine-tune trên data đa dạng hoặc dùng fuzzy matching


Text-to-SQL:
SQL syntax error → Thêm validation layer và error handling


Kết quả sai logic → Schema linking tốt hơn, tăng few-shot examples


Hiệu năng chậm → Quantization (GGUF), caching, hoặc dùng model nhỏ hơn


Độ tin cậy thấp → Hiển thị confidence score và reasoning process




