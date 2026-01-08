"""
SQL Self-Correction Module
Implements iterative error correction without retraining:
1. Execute SQL
2. If error → parse error message
3. Re-prompt model with error context
4. Retry (max 3 attempts)
"""
import re
import duckdb
from typing import Tuple, Optional, List

# ========== COMMON ERROR PATTERNS ==========
ERROR_PATTERNS = {
    # Column not found errors
    r'Table "(\w+)" does not have a column named "(\w+)"': 
        "Column '{col}' does not exist in table '{table}'. Check exact column names in schema.",
    
    # Table not found errors
    r'Referenced table "(\w+)" not found':
        "Table alias '{alias}' not found. Make sure to JOIN the table first.",
    
    # Double prefix errors (r_r_, sr_sr_, etc.)
    r'column "([a-z]{1,2})_\1_\w+"':
        "Double prefix detected. Use single prefix (e.g., r_reason_sk NOT r_r_reason_sk).",
    
    # HAVING with window function
    r'HAVING clause cannot contain window functions':
        "Cannot use RANK()/ROW_NUMBER() in HAVING. Use subquery or CTE instead.",
    
    # GROUP BY missing column
    r'column "(\w+)" must appear in the GROUP BY clause':
        "Column '{col}' must be in GROUP BY or use aggregate function like SUM(), COUNT().",
    
    # Type mismatch
    r'Cannot compare values of type (\w+) and (\w+)':
        "Type mismatch: comparing {type1} with {type2}. Cast or use correct types.",
}

# ========== COLUMN CORRECTIONS ==========
COLUMN_CORRECTIONS = {
    # Wrong table for column
    "c.c_gender": "cd.cd_gender (gender is in customer_demographics)",
    "c.c_credit_rating": "cd.cd_credit_rating (in customer_demographics)",
    "c.c_vehicle_count": "hd.hd_vehicle_count (in household_demographics)",
    "c.c_marital_status": "cd.cd_marital_status (in customer_demographics)",
    "hd.cd_marital_status": "cd.cd_marital_status (in customer_demographics, not household)",
    
    # Wrong column names
    "d_quarter": "d_qoy (quarter of year)",
    "d_weekday": "d_day_name",
    "d_wday": "d_day_name", 
    "d_yrmo": "d_year AND d_moy (use both columns)",
    "inv_quantity": "inv_quantity_on_hand",
    "i_item_name": "i_item_desc or i_product_name",
    
    # Double prefix fixes
    "r_r_reason_sk": "r_reason_sk",
    "sr_sr_return_amt": "sr_return_amt",
    "d_d_date_sk": "d_date_sk",
    "i_i_item_sk": "i_item_sk",
}

# ========== JOIN FIXES ==========
MISSING_JOIN_PATTERNS = {
    r'd\.d_year|d\.d_moy|d\.d_date': 
        ("date_dim", "JOIN date_dim d ON {fact_table}.{date_col} = d.d_date_sk"),
    r'cd\.cd_gender|cd\.cd_marital':
        ("customer_demographics", "JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk"),
    r'hd\.hd_vehicle|hd\.hd_dep':
        ("household_demographics", "JOIN household_demographics hd ON c.c_current_hdemo_sk = hd.hd_demo_sk"),
}


def parse_sql_error(error_msg: str) -> dict:
    """Parse DuckDB error message to extract useful info."""
    result = {
        "error_type": "unknown",
        "details": {},
        "suggestion": None
    }
    
    # Check for column not found
    match = re.search(r'Table "(\w+)" does not have a column named "(\w+)"', error_msg)
    if match:
        result["error_type"] = "column_not_found"
        result["details"] = {"table": match.group(1), "column": match.group(2)}
        
        # Check if we have a correction
        wrong_col = f"{match.group(1)}.{match.group(2)}"
        for pattern, correction in COLUMN_CORRECTIONS.items():
            if match.group(2) in pattern or wrong_col in pattern:
                result["suggestion"] = correction
                break
        return result
    
    # Check for table not found
    match = re.search(r'Referenced table "(\w+)" not found', error_msg)
    if match:
        result["error_type"] = "table_not_found"
        result["details"] = {"alias": match.group(1)}
        result["suggestion"] = f"Add JOIN for table alias '{match.group(1)}'"
        return result
    
    # Check for GROUP BY error
    match = re.search(r'column "(\w+)" must appear in the GROUP BY', error_msg)
    if match:
        result["error_type"] = "group_by_missing"
        result["details"] = {"column": match.group(1)}
        result["suggestion"] = f"Add '{match.group(1)}' to GROUP BY or wrap in aggregate"
        return result
    
    # Check for HAVING with window function
    if "HAVING clause cannot contain window functions" in error_msg:
        result["error_type"] = "having_window"
        result["suggestion"] = "Use subquery: SELECT * FROM (SELECT ..., RANK() ...) WHERE rnk = 1"
        return result
    
    return result


def auto_fix_sql(sql: str) -> str:
    """Apply automatic fixes for common patterns."""
    fixed = sql
    
    # Fix double prefixes: r_r_ -> r_, sr_sr_ -> sr_, d_d_ -> d_
    fixed = re.sub(r'\b([a-z]{1,2})_\1_(\w+)', r'\1_\2', fixed)
    
    # Fix common wrong column names
    replacements = [
        (r'\bd_quarter\b', 'd_qoy'),
        (r'\bd_weekday\b', 'd_day_name'),
        (r'\bd_wday\b', 'd_day_name'),
        (r'\binv_quantity\b(?!_on_hand)', 'inv_quantity_on_hand'),
    ]
    for pattern, replacement in replacements:
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
    
    return fixed


def build_correction_prompt(original_sql: str, error_info: dict, schema_context: str) -> str:
    """Build a prompt for the model to fix the SQL."""
    prompt = f"""The following SQL has an error. Fix it.

ORIGINAL SQL:
{original_sql}

ERROR TYPE: {error_info['error_type']}
ERROR DETAILS: {error_info['details']}
"""
    if error_info.get('suggestion'):
        prompt += f"SUGGESTION: {error_info['suggestion']}\n"
    
    prompt += f"""
SCHEMA REMINDER:
{schema_context}

RULES:
- Fix ONLY the error, keep the rest of the query
- Use exact column names from schema
- Ensure all referenced tables are JOINed

FIXED SQL:"""
    
    return prompt


def self_correct_sql(
    sql: str,
    db_path: str,
    model=None,
    tokenizer=None,
    schema_context: str = "",
    max_attempts: int = 3
) -> Tuple[str, bool, Optional[str]]:
    """
    Self-correction loop: execute SQL, if error, fix and retry.
    
    Returns: (final_sql, success, error_message)
    """
    current_sql = sql
    
    for attempt in range(max_attempts):
        # Try auto-fix first
        current_sql = auto_fix_sql(current_sql)
        
        # Execute
        try:
            con = duckdb.connect(db_path, read_only=True)
            con.execute(current_sql)
            con.close()
            return current_sql, True, None
        except Exception as e:
            error_msg = str(e)
            
            if attempt == max_attempts - 1:
                return current_sql, False, error_msg
            
            # Parse error
            error_info = parse_sql_error(error_msg)
            
            # If we have model, use it to fix
            if model is not None and tokenizer is not None:
                correction_prompt = build_correction_prompt(
                    current_sql, error_info, schema_context
                )
                
                inputs = tokenizer(correction_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                current_sql = tokenizer.decode(gen_ids, skip_special_tokens=True)
                current_sql = extract_sql_from_output(current_sql)
            else:
                # No model, just apply rule-based fixes
                current_sql = apply_error_specific_fix(current_sql, error_info)
    
    return current_sql, False, "Max attempts reached"


def apply_error_specific_fix(sql: str, error_info: dict) -> str:
    """Apply rule-based fix based on error type."""
    fixed = sql
    
    if error_info["error_type"] == "column_not_found":
        table = error_info["details"].get("table", "")
        col = error_info["details"].get("column", "")
        
        # Try to fix common wrong columns
        if col == "c_gender":
            fixed = fixed.replace("c.c_gender", "cd.cd_gender")
            # Add JOIN if missing
            if "customer_demographics" not in fixed:
                fixed = fixed.replace(
                    "FROM customer c",
                    "FROM customer c JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk"
                )
        elif col == "c_credit_rating":
            fixed = fixed.replace("c.c_credit_rating", "cd.cd_credit_rating")
        elif "d_" in col and "date_dim" not in fixed.lower():
            # Missing date_dim JOIN
            pass  # Complex fix, skip for now
    
    elif error_info["error_type"] == "table_not_found":
        alias = error_info["details"].get("alias", "")
        if alias == "d":
            # Need to add date_dim JOIN - find the fact table
            fact_tables = ["store_sales", "web_sales", "catalog_sales"]
            for ft in fact_tables:
                if ft in fixed.lower():
                    # Add JOIN before WHERE
                    fixed = re.sub(
                        r'(FROM\s+\w+\s+\w+)',
                        rf'\1 JOIN date_dim d ON {ft[:2]}.{ft[:2]}_sold_date_sk = d.d_date_sk',
                        fixed,
                        count=1
                    )
                    break
    
    return fixed


def extract_sql_from_output(text: str) -> str:
    """Extract SQL from model output."""
    # Remove markdown code blocks
    text = re.sub(r'^```sql\s*', '', text.strip())
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'```$', '', text)
    
    # Find SELECT or WITH
    match = re.search(r'\b(SELECT|WITH)\b', text, re.IGNORECASE)
    if match:
        text = text[match.start():]
    
    # Cut at first semicolon
    if ';' in text:
        text = text[:text.index(';')+1]
    
    return text.strip()


# ========== STANDALONE TEST ==========
if __name__ == "__main__":
    # Test error parsing
    test_errors = [
        'Binder Error: Table "c" does not have a column named "c_gender"',
        'Binder Error: Referenced table "d" not found!',
        'Binder Error: column "inv_quantity_on_hand" must appear in the GROUP BY clause',
        'Binder Error: HAVING clause cannot contain window functions!',
    ]
    
    print("Testing error parsing:")
    for err in test_errors:
        info = parse_sql_error(err)
        print(f"\nError: {err[:60]}...")
        print(f"Parsed: {info}")
    
    # Test auto-fix
    test_sqls = [
        "SELECT r_r_reason_sk FROM reason",
        "SELECT sr_sr_return_amt FROM store_returns",
        "SELECT d_quarter FROM date_dim",
        "SELECT inv_quantity FROM inventory",
    ]
    
    print("\n\nTesting auto-fix:")
    for sql in test_sqls:
        fixed = auto_fix_sql(sql)
        print(f"Original: {sql}")
        print(f"Fixed:    {fixed}\n")
