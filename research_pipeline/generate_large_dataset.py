import json
import random
import os
import sys
import asyncio
import subprocess
import pandas as pd
import torch
import librosa
import soundfile as sf
import nest_asyncio

# Install edge-tts if missing
try:
    import edge_tts
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "edge-tts"])
    import edge_tts

OUTPUT_DIR = "/kaggle/working/data" if os.path.exists("/kaggle/working") else "data"
TENSOR_DIR = os.path.join(OUTPUT_DIR, "tensors")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

# Data Lists for Randomization
YEARS = [2019, 2020, 2021, 2022, 2023]
MONTHS = range(1, 13)
CATEGORIES = ["Electronics", "Books", "Home", "Clothing", "Sports", "Music", "Toys", "Jewelry"]
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY"]

# Filler Words / Disfluencies
FILLERS = ["ờ", "à", "ừm", "hmmm", "để xem", "kiểu như là", "thì", "ờ thì"]
EMOTIONAL_PREFIXES = ["Cho mình hỏi", "Bạn ơi cho hỏi", "Tính giúp mình", "Tra cứu giúp tôi", "Ê", "Này"]

# Templates
TEMPLATES = [
    ("Tìm top {limit} sản phẩm thuộc danh mục {category} có giá cao hơn {price} đô la", "SELECT i_item_id, i_current_price FROM item WHERE i_current_price > {price} AND i_category = '{category}' ORDER BY i_current_price DESC LIMIT {limit};"),
    ("Đếm số lượng khách hàng ở bang {state}", "SELECT count(*) FROM customer_address WHERE ca_state = '{state}';"),
    ("Tổng doanh thu bán hàng trong năm {year} là bao nhiêu", "SELECT sum(ss_net_paid) FROM store_sales, date_dim WHERE ss_sold_date_sk = d_date_sk AND d_year = {year};"),
    ("Liệt kê {limit} mặt hàng bán chạy nhất tháng {month}", "SELECT i_item_desc, COUNT(*) FROM store_sales, item, date_dim WHERE ss_item_sk = i_item_sk AND ss_sold_date_sk = d_date_sk AND d_moy = {month} GROUP BY i_item_desc ORDER BY 2 DESC LIMIT {limit};"),
    ("Khách hàng nào đã chi tiêu nhiều nhất trong năm {year}", "SELECT c_first_name, c_last_name, SUM(ss_net_paid) FROM customer, store_sales, date_dim WHERE c_customer_sk = ss_customer_sk AND ss_sold_date_sk = d_date_sk AND d_year = {year} GROUP BY c_first_name, c_last_name ORDER BY 3 DESC LIMIT 1;"),
    ("Tìm danh sách các khách hàng sống tại {state} mua hàng trong năm {year}", "SELECT distinct c_first_name, c_last_name FROM customer, customer_address, store_sales, date_dim WHERE c_current_addr_sk = ca_address_sk AND ca_state = '{state}' AND c_customer_sk = ss_customer_sk AND ss_sold_date_sk = d_date_sk AND d_year = {year};"),
    ("Tính giá trị trung bình của các đơn hàng trong danh mục {category}", "SELECT AVG(ss_net_paid) FROM store_sales, item WHERE ss_item_sk = i_item_sk AND i_category = '{category}';"),
    ("Hiển thị {limit} cửa hàng có doanh số cao nhất năm {year}", "SELECT s_store_name, SUM(ss_net_paid) FROM store, store_sales, date_dim WHERE s_store_sk = ss_store_sk AND ss_sold_date_sk = d_date_sk AND d_year = {year} GROUP BY s_store_name ORDER BY 2 DESC LIMIT {limit};"),
    ("Số lượng hàng tồn kho của sản phẩm thuộc nhóm {category} là bao nhiêu", "SELECT SUM(inv_quantity_on_hand) FROM inventory, item WHERE inv_item_sk = i_item_sk AND i_category = '{category}';"),
    ("Tìm các mặt hàng {category} có giá thấp hơn {price} đô la", "SELECT i_item_desc FROM item WHERE i_category = '{category}' AND i_current_price < {price};")
]

def inject_disfluencies(text):
    if random.random() < 0.3:
        text = f"{random.choice(EMOTIONAL_PREFIXES)} {text.lower()}"
    words = text.split()
    if len(words) > 4 and random.random() < 0.4:
        idx = random.randint(1, len(words) - 2)
        words.insert(idx, f"... {random.choice(FILLERS)} ...")
        text = " ".join(words)
    return text.replace("... ...", "...")

def generate_text_data(num_samples=100):
    data = []
    for i in range(1, num_samples + 1):
        tmpl_q, tmpl_s = random.choice(TEMPLATES)
        ctx = {
            "limit": random.choice([3, 5, 10, 20, 50]),
            "category": random.choice(CATEGORIES),
            "price": random.choice([10, 50, 100, 200, 500, 1000]),
            "state": random.choice(STATES),
            "year": random.choice(YEARS),
            "month": random.choice(MONTHS)
        }
        question = tmpl_q.format(**ctx)
        sql = tmpl_s.format(**ctx)
        question_natural = inject_disfluencies(question)
        
        data.append({
            "id": f"q{i}",
            "text": question_natural,
            "original_text": question,
            "sql": sql
        })
    return data

async def generate_audio_and_tensors(queries):
    os.makedirs(TENSOR_DIR, exist_ok=True)
    
    VOICES = ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"]
    AUGMENTATIONS = [
        {"rate": "+0%", "pitch": "+0Hz", "suffix": ""},
        {"rate": "+10%", "suffix": "_fast"},
        {"rate": "-10%", "suffix": "_slow"},
        {"pitch": "+5Hz", "suffix": "_high"},
    ]
    
    metadata = []
    print(f"Generating {len(queries)} samples x {len(VOICES)*len(AUGMENTATIONS)} variations...")
    
    for i, q in enumerate(queries):
        for voice in VOICES:
            voice_gender = "male" if "NamMinh" in voice else "female"
            for aug in AUGMENTATIONS:
                aug_suffix = aug["suffix"]
                file_id = f"{q['id']}_{voice_gender}{aug_suffix}"
                
                # Check if tensor exists
                tensor_filename = f"{file_id}.pt"
                tensor_path = os.path.join(TENSOR_DIR, tensor_filename)
                
                if not os.path.exists(tensor_path):
                    # 1. Generate Temp MP3
                    mp3_path = f"/tmp/{file_id}.mp3"
                    communicate = edge_tts.Communicate(
                        q['text'], voice, 
                        rate=aug.get("rate", "+0%"), pitch=aug.get("pitch", "+0Hz")
                    )
                    await communicate.save(mp3_path)
                    
                    # 2. Convert to Tensor (16kHz)
                    arr, _ = librosa.load(mp3_path, sr=16000)
                    torch.save(torch.from_numpy(arr), tensor_path)
                    
                    # 3. Cleanup
                    if os.path.exists(mp3_path):
                        os.remove(mp3_path)
                
                metadata.append({
                    "id": file_id,
                    "query_id": q['id'],
                    "text": q['text'],
                    "voice": voice_gender,
                    "augmentation": aug_suffix,
                    "tensor_filename": tensor_filename,
                    "sql": q['sql']
                })
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(queries)} queries.")

    # Save Metadata
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_FILE, index=False)
    print(f"✅ Generated {len(metadata)} tensors in {TENSOR_DIR}")
    print(f"✅ Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    queries = generate_text_data(100)
    
    loop = asyncio.get_event_loop()
    if loop.is_running():
        nest_asyncio.apply()
        loop.run_until_complete(generate_audio_and_tensors(queries))
    else:
        asyncio.run(generate_audio_and_tensors(queries))
