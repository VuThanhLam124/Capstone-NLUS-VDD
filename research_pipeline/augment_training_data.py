#!/usr/bin/env python3
"""
Data Augmentation for Text-to-SQL Training

Ky thuat augment:
1. Paraphrase questions: thay doi cach dat cau hoi
2. Add noise/typos: mo phong loi danh may cua nguoi dung
3. Synonyms: thay the cac tu dong nghia
4. Column/Value variation: thay doi gia tri trong SQL

Output: train_augmented.jsonl
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)

# ========== AUGMENTATION PATTERNS ==========

# Cac cau hoi dong nghia
QUESTION_PARAPHRASES = {
    "liệt kê": ["cho tôi xem", "hiển thị", "tìm giúp tôi", "lấy danh sách", "truy vấn"],
    "tìm": ["lọc ra", "tìm kiếm", "cho biết", "xác định"],
    "bao nhiêu": ["có bao nhiêu", "tổng số", "đếm xem có", "số lượng"],
    "doanh thu": ["doanh số", "tổng bán hàng", "revenue", "lợi nhuận bán hàng"],
    "tổng": ["cộng tất cả", "sum", "tính tổng"],
    "khách hàng": ["người mua", "customer", "người dùng"],
    "sản phẩm": ["mặt hàng", "hàng hóa", "items", "món đồ"],
    "mua": ["đặt hàng", "order", "giao dịch"],
    "năm": ["trong năm", "vào năm"],
    "quý": ["trong quý", "quý số"],
    "tại": ["ở", "thuộc", "nằm ở"],
    "cao nhất": ["lớn nhất", "top", "đứng đầu", "nhiều nhất"],
    "thấp nhất": ["nhỏ nhất", "ít nhất", "cuối bảng"],
}

# Cac states de thay the
US_STATES = ["CA", "TX", "NY", "FL", "IL", "GA", "NC", "VA", "MO", "KY", "IA", "KS"]

# Cac nam trong TPC-DS
YEARS = [1998, 1999, 2000, 2001, 2002]

# Cac quy
QUARTERS = [1, 2, 3, 4]

# Cac category trong TPC-DS
CATEGORIES = ["Books", "Children", "Electronics", "Home", "Jewelry", "Men", "Music", "Shoes", "Sports", "Women"]

# Education status
EDUCATION_STATUS = ["Primary", "Secondary", "College", "2 yr Degree", "4 yr Degree", "Advanced Degree", "Unknown"]


def paraphrase_question(question: str) -> str:
    """Thay the cac cum tu bang dong nghia"""
    result = question
    
    # Chon ngau nhien 1-2 pattern de thay the
    patterns_to_apply = random.sample(
        list(QUESTION_PARAPHRASES.keys()), 
        min(2, len(QUESTION_PARAPHRASES))
    )
    
    for pattern in patterns_to_apply:
        if pattern.lower() in result.lower():
            replacement = random.choice(QUESTION_PARAPHRASES[pattern])
            # Case insensitive replace
            result = re.sub(
                re.escape(pattern), 
                replacement, 
                result, 
                flags=re.IGNORECASE,
                count=1
            )
    
    return result


def add_typos(question: str, typo_rate: float = 0.05) -> str:
    """Them loi danh may ngau nhien (mo phong nguoi dung thuc)"""
    # Chi ap dung voi xac suat thap
    if random.random() > 0.3:
        return question
    
    chars = list(question)
    num_typos = max(1, int(len(chars) * typo_rate))
    
    for _ in range(num_typos):
        idx = random.randint(0, len(chars) - 1)
        if chars[idx].isalpha():
            # Typo types: swap, double, delete
            typo_type = random.choice(["swap", "double", "delete"])
            if typo_type == "swap" and idx < len(chars) - 1:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif typo_type == "double":
                chars.insert(idx, chars[idx])
            elif typo_type == "delete":
                chars[idx] = ""
    
    return "".join(chars)


def substitute_values(question: str, sql: str) -> tuple:
    """Thay the cac gia tri trong cau hoi va SQL"""
    new_question = question
    new_sql = sql
    
    # Thay state
    for old_state in US_STATES:
        if f"'{old_state}'" in sql or f" {old_state}" in question:
            new_state = random.choice([s for s in US_STATES if s != old_state])
            new_question = new_question.replace(old_state, new_state)
            new_sql = new_sql.replace(f"'{old_state}'", f"'{new_state}'")
            break
    
    # Thay year
    year_match = re.search(r'\b(199[89]|200[012])\b', question)
    if year_match:
        old_year = year_match.group(1)
        new_year = str(random.choice([y for y in YEARS if str(y) != old_year]))
        new_question = new_question.replace(old_year, new_year)
        new_sql = new_sql.replace(old_year, new_year)
    
    # Thay quarter
    for qoy in QUARTERS:
        pattern = f"quý {qoy}"
        if pattern in question.lower():
            new_qoy = random.choice([q for q in QUARTERS if q != qoy])
            new_question = re.sub(rf'quý\s*{qoy}', f'quý {new_qoy}', new_question, flags=re.IGNORECASE)
            new_sql = re.sub(rf"d\.d_qoy\s*=\s*{qoy}", f"d.d_qoy = {new_qoy}", new_sql)
            new_sql = re.sub(rf"d_qoy\s*=\s*{qoy}", f"d_qoy = {new_qoy}", new_sql)
            break
    
    return new_question, new_sql


def substitute_category(question: str, sql: str) -> tuple:
    """Thay the category trong cau hoi va SQL"""
    new_question = question
    new_sql = sql
    
    for old_cat in CATEGORIES:
        if f"'{old_cat}'" in sql or old_cat.lower() in question.lower():
            new_cat = random.choice([c for c in CATEGORIES if c != old_cat])
            new_question = re.sub(old_cat, new_cat, new_question, flags=re.IGNORECASE)
            new_sql = new_sql.replace(f"'{old_cat}'", f"'{new_cat}'")
            break
    
    return new_question, new_sql


def augment_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Augment mot sample thanh nhieu samples"""
    augmented = []
    messages = sample.get("messages", [])
    
    if len(messages) < 3:  # Phai co system, user, assistant
        return []
    
    system_msg = messages[0]["content"]
    user_msg = messages[1]["content"]
    assistant_sql = messages[2]["content"]
    
    # Tach question tu user message
    question_match = re.search(r'QUESTION:\s*\n?(.+?)\n\nSQL:', user_msg, re.DOTALL)
    if not question_match:
        return []
    
    original_question = question_match.group(1).strip()
    
    # 1. Paraphrase only
    para_question = paraphrase_question(original_question)
    if para_question != original_question:
        new_user_msg = user_msg.replace(f"QUESTION:\n{original_question}", f"QUESTION:\n{para_question}")
        augmented.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": new_user_msg},
                {"role": "assistant", "content": assistant_sql}
            ]
        })
    
    # 2. Value substitution (state, year, quarter)
    sub_question, sub_sql = substitute_values(original_question, assistant_sql)
    if sub_question != original_question:
        new_user_msg = user_msg.replace(f"QUESTION:\n{original_question}", f"QUESTION:\n{sub_question}")
        augmented.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": new_user_msg},
                {"role": "assistant", "content": sub_sql}
            ]
        })
    
    # 3. Category substitution
    cat_question, cat_sql = substitute_category(original_question, assistant_sql)
    if cat_question != original_question:
        new_user_msg = user_msg.replace(f"QUESTION:\n{original_question}", f"QUESTION:\n{cat_question}")
        augmented.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": new_user_msg},
                {"role": "assistant", "content": cat_sql}
            ]
        })
    
    # 4. Typo version (mo phong loi danh may)
    typo_question = add_typos(original_question)
    if typo_question != original_question:
        new_user_msg = user_msg.replace(f"QUESTION:\n{original_question}", f"QUESTION:\n{typo_question}")
        augmented.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": new_user_msg},
                {"role": "assistant", "content": assistant_sql}  # SQL van giu nguyen
            ]
        })
    
    return augmented


def main():
    input_path = Path("research_pipeline/datasets/train_schema_aware.jsonl")
    output_path = Path("research_pipeline/datasets/train_augmented.jsonl")
    
    print(f"Reading from: {input_path}")
    
    original_samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                original_samples.append(json.loads(line))
    
    print(f"Original samples: {len(original_samples)}")
    
    # Augment
    augmented_samples = []
    for i, sample in enumerate(original_samples):
        # Giu lai sample goc
        augmented_samples.append(sample)
        
        # Them cac ban augmented
        new_samples = augment_sample(sample)
        augmented_samples.extend(new_samples)
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(original_samples)} samples, total: {len(augmented_samples)}")
    
    # Shuffle
    random.shuffle(augmented_samples)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nAugmented samples: {len(augmented_samples)}")
    print(f"Saved to: {output_path}")
    print(f"Increase: {len(augmented_samples) - len(original_samples)} samples ({100*(len(augmented_samples)/len(original_samples) - 1):.1f}% more)")


if __name__ == "__main__":
    main()
