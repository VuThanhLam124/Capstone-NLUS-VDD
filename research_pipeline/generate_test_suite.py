import json
import csv
import asyncio
import os
import edge_tts

QUERIES_FILE = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/test_queries.json"
AUDIO_DIR = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/audio"
OUTPUT_CSV = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/test_suite.csv"

# Voice for TTS (Vietnamese or English? The questions are English in JSON implies English)
# User context says "Vietnamese is primary language" but TPC-DS is often English.
# The user's prompt was in Vietnamese but the domain is standard E-commerce.
# I'll stick to English for the standard benchmark questions as TPC-DS is standard, 
# but I can add a few Vietnamese ones if needed. For now, English to match TPC-DS.
VOICE = "en-US-ChristopherNeural"

async def generate_audio(text, output_file):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_file)

async def main():
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
        
    with open(QUERIES_FILE, 'r') as f:
        data = json.load(f)
        
    print(f"Generating audio for {len(data)} queries...")
    
    rows = []
    
    for item in data:
        q_id = item['id']
        text = item['question']
        sql = item['sql']
        
        audio_filename = f"{q_id}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        print(f"Processing {q_id}: {text}")
        await generate_audio(text, audio_path)
        
        rows.append({
            "id": q_id,
            "audio_path": audio_path,
            "question_ground_truth": text,
            "sql_ground_truth": sql
        })
        
    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['id', 'audio_path', 'question_ground_truth', 'sql_ground_truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            
    print(f"Done! Test suite saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
