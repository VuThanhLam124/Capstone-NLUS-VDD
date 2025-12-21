import csv
import time
import duckdb
import jiwer
import difflib
import os
import random
# import speech_recognition as sr  # Uncomment if real ASR is needed
# import openai # Uncomment if real LLM is needed

# CONFIG
TEST_SUITE_PATH = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/test_suite.csv"
DB_PATH = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/data/ecommerce_dw.duckdb"
RESULTS_PATH = "/home/ubuntu/DataScience/Capstone-NLUS-VDD/research_pipeline/benchmark_results.csv"

# SIMULATION MODES (Set to False to use specific Real Models)
SIMULATE_ASR = True
SIMULATE_LLM = True

class BenchmarkRunner:
    def __init__(self):
        self.con = duckdb.connect(DB_PATH, read_only=True)
        # Load Vector DB / Cache if needed (from transition_demo.py)
        
    def run_asr(self, audio_path, ground_truth_text):
        """
        Simulate ASR or use real model.
        Returns: transcribed_text, latency
        """
        start = time.time()
        
        if SIMULATE_ASR:
            # Simulate some errors (perturbation)
            # 5-10% Error rate simulation
            text = ground_truth_text
            if random.random() > 0.7:
                text = text.replace("items", "item").replace("calculated", "")
            time.sleep(0.5)
            transcribed = text
        else:
            # Real ASR Logic Here
            pass
            
        latency = time.time() - start
        return transcribed, latency

    def run_text_to_sql(self, text, ground_truth_sql):
        """
        Simulate Text-to-SQL or use LLM.
        Returns: generated_sql, latency
        """
        start = time.time()
        
        if SIMULATE_LLM:
            # Simulate a good model but not perfect
            # For this demo, let's just return the Ground Truth SQL 
            # (assuming perfect LLM for now to test pipeline)
            # Or maybe introduce a small error?
            sql = ground_truth_sql
            time.sleep(1.0)
        else:
            # Call OpenAI / Local LLM
            pass
            
        latency = time.time() - start
        return sql, latency

    def execute_sql(self, sql):
        """
        Execute SQL on DuckDB
        Returns: result_set (list of tuples), error_message
        """
        try:
            # Using read_only connection
            # Note: TPC-DS in DuckDB might need TPC-DS extension loaded
            # but since we generated verified data, standard SQL should work 
            # if tables exist.
            res = self.con.sql(sql).fetchall()
            return res, None
        except Exception as e:
            return None, str(e)

    def calculate_metrics(self, row, trans_text, gen_sql, exec_res_gen, exec_res_gt):
        # 1. WER (Word Error Rate)
        wer = jiwer.wer(row['question_ground_truth'], trans_text)
        
        # 2. SQL Execution Match (EM)
        # Compare result sets (ignore order if appropriate, but list comparison is strict)
        # Using string representation for simple comparison
        exec_match = (str(exec_res_gen) == str(exec_res_gt))
        
        return {
            "id": row['id'],
            "wer": wer,
            "exec_match": exec_match,
            "transcription": trans_text,
            "generated_sql": gen_sql,
            "result_summary": str(exec_res_gen)[:50] + "..." if exec_res_gen else "Error"
        }

    def run(self):
        results = []
        
        # 1. Load Test Suite
        with open(TEST_SUITE_PATH, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            
        print(f"Starting Benchmark on {len(data)} items...")
            
        for row in data:
            print(f"Processing {row['id']}...")
            
            # Step A: Ground Truth Execution (to compare against)
            gt_res, gt_err = self.execute_sql(row['sql_ground_truth'])
            if gt_err:
                print(f"  [WARNING] Ground Truth SQL failed: {gt_err}")
                continue

            # Step B: ASR
            trans_text, asr_time = self.run_asr(row['audio_path'], row['question_ground_truth'])
            
            # Step C: Text-to-SQL
            # Note: In a real system, we would pass 'trans_text', not the GT query.
            gen_sql, sql_time = self.run_text_to_sql(trans_text, row['sql_ground_truth'])
            
            # Step D: Execution
            gen_res, gen_err = self.execute_sql(gen_sql)
            
            # Step E: Metrics
            metrics = self.calculate_metrics(row, trans_text, gen_sql, gen_res, gt_res)
            metrics['asr_latency'] = asr_time
            metrics['sql_latency'] = sql_time
            
            results.append(metrics)
            print(f"  -> WER: {metrics['wer']:.2f}, ExecMatch: {metrics['exec_match']}")
            
        # Save Results
        keys = results[0].keys()
        with open(RESULTS_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Benchmark finished. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
