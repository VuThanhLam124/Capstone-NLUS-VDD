#!/usr/bin/env python3
"""
Analyze difficulty of test datasets
"""
import pandas as pd
import re
from collections import Counter

def analyze_sql_complexity(sql: str) -> dict:
    """Analyze complexity metrics of SQL query"""
    sql_lower = sql.lower()
    
    # Count tables (FROM and JOIN)
    tables = len(re.findall(r'\b(?:from|join)\s+(\w+)', sql_lower))
    
    # Count JOINs
    joins = len(re.findall(r'\bjoin\b', sql_lower))
    
    # Check for aggregations
    has_agg = bool(re.search(r'\b(?:sum|count|avg|min|max|group\s+by)\b', sql_lower))
    
    # Check for subqueries
    subqueries = len(re.findall(r'\bselect\b', sql_lower)) - 1  # -1 for main SELECT
    
    # Check for WHERE conditions
    where_conditions = len(re.findall(r'\bwhere\b', sql_lower))
    
    # Count ORDER BY
    has_order = bool(re.search(r'\border\s+by\b', sql_lower))
    
    # Count LIMIT
    has_limit = bool(re.search(r'\blimit\b', sql_lower))
    
    # SQL length
    sql_length = len(sql)
    
    return {
        "tables": tables,
        "joins": joins,
        "has_aggregation": has_agg,
        "subqueries": subqueries,
        "where_conditions": where_conditions,
        "has_order_by": has_order,
        "has_limit": has_limit,
        "sql_length": sql_length
    }


def analyze_dataset(csv_path: str, name: str):
    """Analyze test dataset"""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    
    # Get SQL column
    sql_col = "SQL Ground Truth" if "SQL Ground Truth" in df.columns else "sql"
    q_col = "Transcription" if "Transcription" in df.columns else "question"
    
    print(f"\nTotal samples: {len(df)}")
    
    # Analyze each query
    all_metrics = []
    for idx, row in df.iterrows():
        sql = row[sql_col]
        metrics = analyze_sql_complexity(sql)
        all_metrics.append(metrics)
    
    # Aggregate statistics
    metrics_df = pd.DataFrame(all_metrics)
    
    print(f"\n--- SQL Complexity Statistics ---")
    print(f"Average tables per query: {metrics_df['tables'].mean():.1f}")
    print(f"Average JOINs per query: {metrics_df['joins'].mean():.1f}")
    print(f"Queries with aggregation: {metrics_df['has_aggregation'].sum()} ({metrics_df['has_aggregation'].mean()*100:.1f}%)")
    print(f"Queries with subqueries: {metrics_df['subqueries'].sum()} ({(metrics_df['subqueries'] > 0).mean()*100:.1f}%)")
    print(f"Queries with WHERE: {metrics_df['where_conditions'].sum()} ({(metrics_df['where_conditions'] > 0).mean()*100:.1f}%)")
    print(f"Queries with ORDER BY: {metrics_df['has_order_by'].sum()} ({metrics_df['has_order_by'].mean()*100:.1f}%)")
    print(f"Average SQL length: {metrics_df['sql_length'].mean():.0f} chars")
    
    print(f"\n--- Complexity Distribution ---")
    print(f"Tables used:")
    print(metrics_df['tables'].value_counts().sort_index())
    
    print(f"\nJOINs count:")
    print(metrics_df['joins'].value_counts().sort_index())
    
    # Classify difficulty
    easy_count = ((metrics_df['tables'] <= 2) & (metrics_df['joins'] <= 1) & (metrics_df['subqueries'] == 0)).sum()
    medium_count = ((metrics_df['tables'] <= 4) & (metrics_df['joins'] <= 3) & (metrics_df['subqueries'] <= 1)).sum() - easy_count
    hard_count = len(metrics_df) - easy_count - medium_count
    
    print(f"\n--- Difficulty Classification ---")
    print(f"Easy (≤2 tables, ≤1 JOIN, no subquery): {easy_count} ({easy_count/len(df)*100:.1f}%)")
    print(f"Medium (≤4 tables, ≤3 JOINs, ≤1 subquery): {medium_count} ({medium_count/len(df)*100:.1f}%)")
    print(f"Hard (>4 tables OR >3 JOINs OR >1 subquery): {hard_count} ({hard_count/len(df)*100:.1f}%)")
    
    # Sample easy and hard queries
    print(f"\n--- Sample Easy Query ---")
    easy_idx = metrics_df[(metrics_df['tables'] <= 2) & (metrics_df['joins'] <= 1)].index[0]
    print(f"Question: {df.iloc[easy_idx][q_col]}")
    print(f"SQL: {df.iloc[easy_idx][sql_col][:150]}...")
    
    print(f"\n--- Sample Hard Query ---")
    hard_idx = metrics_df[(metrics_df['tables'] >= 4) | (metrics_df['joins'] >= 3)].index[0] if ((metrics_df['tables'] >= 4) | (metrics_df['joins'] >= 3)).any() else 0
    print(f"Question: {df.iloc[hard_idx][q_col]}")
    print(f"SQL: {df.iloc[hard_idx][sql_col][:150]}...")
    
    return metrics_df


# Analyze both datasets
easy_metrics = analyze_dataset("research_pipeline/datasets/test_easy.csv", "test_easy.csv")
test_metrics = analyze_dataset("research_pipeline/datasets/test.csv", "test.csv (full)")

print(f"\n{'='*60}")
print("COMPARISON: Easy vs Full Test")
print(f"{'='*60}")
print(f"{'Metric':<30} {'Easy':<15} {'Full Test'}")
print("-" * 60)
print(f"{'Avg Tables':<30} {easy_metrics['tables'].mean():.2f}{'':<13} {test_metrics['tables'].mean():.2f}")
print(f"{'Avg JOINs':<30} {easy_metrics['joins'].mean():.2f}{'':<13} {test_metrics['joins'].mean():.2f}")
print(f"{'With Aggregation %':<30} {easy_metrics['has_aggregation'].mean()*100:.1f}%{'':<11} {test_metrics['has_aggregation'].mean()*100:.1f}%")
print(f"{'With Subqueries %':<30} {(easy_metrics['subqueries']>0).mean()*100:.1f}%{'':<11} {(test_metrics['subqueries']>0).mean()*100:.1f}%")
print(f"{'Avg SQL Length':<30} {easy_metrics['sql_length'].mean():.0f}{'':<12} {test_metrics['sql_length'].mean():.0f}")
