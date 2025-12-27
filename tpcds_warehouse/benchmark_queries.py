#!/usr/bin/env python3
"""
TPC-DS Query Benchmark Script
============================
Measures query latency for sample TPC-DS queries via Trino.

Usage: python3 benchmark_queries.py
"""

import subprocess
import time
import json
import csv
from datetime import datetime
from pathlib import Path

# Configuration
TRINO_CONTAINER = "trino"
CATALOG = "tpcds"  # Built-in TPC-DS catalog in Trino
SCHEMA = "sf1"     # Scale factor 1
OUTPUT_FILE = Path(__file__).parent / "benchmark_results_trino.csv"

# Sample TPC-DS queries for benchmarking
BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "name": "Simple Count",
        "sql": "SELECT COUNT(*) FROM store_sales",
        "category": "aggregation"
    },
    {
        "id": "Q2",
        "name": "Date Filter Count",
        "sql": """
            SELECT COUNT(*) 
            FROM store_sales ss
            JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
            WHERE d.d_year = 2000
        """,
        "category": "filter"
    },
    {
        "id": "Q3",
        "name": "Top Items by Revenue",
        "sql": """
            SELECT 
                i.i_item_id,
                i.i_item_desc,
                SUM(ss.ss_sales_price) as total_revenue
            FROM store_sales ss
            JOIN item i ON ss.ss_item_sk = i.i_item_sk
            GROUP BY i.i_item_id, i.i_item_desc
            ORDER BY total_revenue DESC
            LIMIT 10
        """,
        "category": "aggregation"
    },
    {
        "id": "Q4",
        "name": "Sales by Store and Year",
        "sql": """
            SELECT 
                s.s_store_name,
                d.d_year,
                SUM(ss.ss_net_profit) as total_profit
            FROM store_sales ss
            JOIN store s ON ss.ss_store_sk = s.s_store_sk
            JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
            GROUP BY s.s_store_name, d.d_year
            ORDER BY s.s_store_name, d.d_year
        """,
        "category": "aggregation"
    },
    {
        "id": "Q5",
        "name": "Customer Spending Analysis",
        "sql": """
            SELECT 
                c.c_customer_id,
                c.c_first_name,
                c.c_last_name,
                SUM(ss.ss_net_paid) as total_spent
            FROM store_sales ss
            JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
            GROUP BY c.c_customer_id, c.c_first_name, c.c_last_name
            ORDER BY total_spent DESC
            LIMIT 20
        """,
        "category": "aggregation"
    },
    {
        "id": "Q6",
        "name": "Inventory Analysis",
        "sql": """
            SELECT 
                COUNT(*) as record_count,
                AVG(inv_quantity_on_hand) as avg_quantity,
                MAX(inv_quantity_on_hand) as max_quantity
            FROM inventory
        """,
        "category": "aggregation"
    },
    {
        "id": "Q7",
        "name": "Multi-Channel Sales",
        "sql": """
            SELECT 'store' as channel, COUNT(*) as transactions FROM store_sales
            UNION ALL
            SELECT 'catalog' as channel, COUNT(*) as transactions FROM catalog_sales
            UNION ALL
            SELECT 'web' as channel, COUNT(*) as transactions FROM web_sales
        """,
        "category": "union"
    },
    {
        "id": "Q8",
        "name": "Complex Join",
        "sql": """
            SELECT 
                d.d_year,
                d.d_moy,
                s.s_store_name,
                i.i_category,
                SUM(ss.ss_sales_price) as monthly_sales
            FROM store_sales ss
            JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
            JOIN store s ON ss.ss_store_sk = s.s_store_sk
            JOIN item i ON ss.ss_item_sk = i.i_item_sk
            WHERE d.d_year BETWEEN 1999 AND 2001
            GROUP BY d.d_year, d.d_moy, s.s_store_name, i.i_category
            ORDER BY d.d_year, d.d_moy, monthly_sales DESC
            LIMIT 50
        """,
        "category": "complex"
    }
]


def run_trino_query(sql: str) -> tuple:
    """Execute query and return (success, output, error, duration_ms)."""
    # Clean up SQL
    clean_sql = " ".join(sql.split()).replace('"', '\\"')
    
    # Use 2>&1 to merge stderr into stdout (Trino CLI outputs warnings to stderr)
    cmd = f'docker exec {TRINO_CONTAINER} trino --catalog {CATALOG} --schema {SCHEMA} --execute "{clean_sql}" 2>&1'
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration_ms = (time.time() - start_time) * 1000
    
    output = result.stdout
    
    # Check for real errors (not just terminal warnings)
    # Trino CLI always outputs a warning about terminal to stderr, ignore it
    has_error = False
    error_msg = ""
    
    if result.returncode != 0:
        # Check if output contains actual error indicators
        if "Query failed" in output or "Error running command" in output or "does not exist" in output:
            has_error = True
            error_msg = output
    
    return not has_error, output, error_msg, duration_ms


def run_benchmark(iterations: int = 3):
    """Run benchmark queries."""
    results = []
    
    print("=" * 70)
    print("📊 TPC-DS Query Benchmark via Trino")
    print("=" * 70)
    print(f"Iterations per query: {iterations}")
    print(f"Total queries: {len(BENCHMARK_QUERIES)}")
    print("-" * 70)
    
    for query in BENCHMARK_QUERIES:
        print(f"\n🔹 {query['id']}: {query['name']}")
        print(f"   Category: {query['category']}")
        
        latencies = []
        success_count = 0
        
        for i in range(iterations):
            success, output, error, duration = run_trino_query(query['sql'])
            
            if success:
                latencies.append(duration)
                success_count += 1
                print(f"   Run {i+1}: {duration:.2f} ms ✓")
            else:
                print(f"   Run {i+1}: FAILED ✗")
                if error:
                    print(f"          Error: {error[:100]}...")
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            results.append({
                "query_id": query['id'],
                "query_name": query['name'],
                "category": query['category'],
                "iterations": iterations,
                "success_count": success_count,
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min_latency, 2),
                "max_latency_ms": round(max_latency, 2),
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"   📈 Avg: {avg_latency:.2f} ms | Min: {min_latency:.2f} ms | Max: {max_latency:.2f} ms")
        else:
            results.append({
                "query_id": query['id'],
                "query_name": query['name'],
                "category": query['category'],
                "iterations": iterations,
                "success_count": 0,
                "avg_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "timestamp": datetime.now().isoformat()
            })
    
    return results


def save_results(results: list):
    """Save benchmark results to CSV."""
    if not results:
        return
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")


def print_summary(results: list):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r['avg_latency_ms'] is not None]
    
    if successful:
        avg_all = sum(r['avg_latency_ms'] for r in successful) / len(successful)
        fastest = min(successful, key=lambda x: x['min_latency_ms'])
        slowest = max(successful, key=lambda x: x['max_latency_ms'])
        
        print(f"\n✅ Successful queries: {len(successful)}/{len(results)}")
        print(f"⏱️  Average latency: {avg_all:.2f} ms")
        print(f"🚀 Fastest query: {fastest['query_id']} ({fastest['min_latency_ms']:.2f} ms)")
        print(f"🐢 Slowest query: {slowest['query_id']} ({slowest['max_latency_ms']:.2f} ms)")
        
        # Category breakdown
        print("\n📊 By Category:")
        categories = {}
        for r in successful:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['avg_latency_ms'])
        
        for cat, latencies in categories.items():
            avg = sum(latencies) / len(latencies)
            print(f"   {cat}: {avg:.2f} ms avg ({len(latencies)} queries)")
    else:
        print("❌ No successful queries")


def check_trino():
    """Check if Trino is accessible."""
    print("🔌 Checking Trino connection...")
    success, _, _, _ = run_trino_query("SELECT 1")
    if not success:
        print("   ✗ Cannot connect to Trino")
        print("   Make sure the stack is running: docker-compose up -d")
        return False
    print("   ✓ Trino is ready")
    return True


def main():
    if not check_trino():
        return 1
    
    # Run benchmark
    results = run_benchmark(iterations=3)
    
    # Save results
    save_results(results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
