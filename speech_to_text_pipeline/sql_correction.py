import difflib
import re
import unicodedata

SQL_KEYWORDS = [
    "SELECT",
    "FROM",
    "WHERE",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "FULL",
    "CROSS",
    "GROUP",
    "ORDER",
    "BY",
    "LIMIT",
    "TOP",
    "DISTINCT",
    "UNION",
    "ALL",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "TABLE",
    "ALTER",
    "DROP",
    "VALUES",
    "INTO",
    "AS",
    "ON",
    "HAVING",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "SUM",
    "AND",
    "OR",
    "NOT",
    "LIKE",
    "IN",
    "IS",
    "NULL",
    "BETWEEN",
]

SQL_ALIAS_MAP = {
    "select": "SELECT",
    "sellect": "SELECT",
    "selec": "SELECT",
    "selet": "SELECT",
    "slect": "SELECT",
    "selct": "SELECT",
    "selech": "SELECT",
    "solech": "SELECT",
    "from": "FROM",
    "form": "FROM",
    "frum": "FROM",
    "fram": "FROM",
    "where": "WHERE",
    "wher": "WHERE",
    "were": "WHERE",
    "join": "JOIN",
    "zoin": "JOIN",
    "goin": "JOIN",
    "left": "LEFT",
    "right": "RIGHT",
    "inner": "INNER",
    "outer": "OUTER",
    "full": "FULL",
    "cross": "CROSS",
    "group": "GROUP",
    "groub": "GROUP",
    "order": "ORDER",
    "oder": "ORDER",
    "by": "BY",
    "bai": "BY",
    "top": "TOP",
    "limit": "LIMIT",
    "distinct": "DISTINCT",
    "distinc": "DISTINCT",
    "union": "UNION",
    "insert": "INSERT",
    "update": "UPDATE",
    "delete": "DELETE",
    "create": "CREATE",
    "table": "TABLE",
    "alter": "ALTER",
    "drop": "DROP",
    "values": "VALUES",
    "into": "INTO",
    "as": "AS",
    "on": "ON",
    "having": "HAVING",
    "count": "COUNT",
    "avg": "AVG",
    "min": "MIN",
    "max": "MAX",
    "sum": "SUM",
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "like": "LIKE",
    "in": "IN",
    "is": "IS",
    "null": "NULL",
    "between": "BETWEEN",
    "groupby": "GROUP BY",
    "orderby": "ORDER BY",
    "leftjoin": "LEFT JOIN",
    "rightjoin": "RIGHT JOIN",
    "innerjoin": "INNER JOIN",
    "outerjoin": "OUTER JOIN",
    "fulljoin": "FULL JOIN",
    "fullouterjoin": "FULL OUTER JOIN",
    "crossjoin": "CROSS JOIN",
    "unionall": "UNION ALL",
}

FUZZY_MIN_LEN = 4
FUZZY_THRESHOLD = 0.84
MERGE_THRESHOLD = 0.88

_WORD_RE = re.compile(r"^(\W*)(\w+)(\W*)$", re.UNICODE)


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_token(token):
    token = strip_accents(token.lower())
    token = re.sub(r"[^0-9a-z_]+", "", token)
    return token


SQL_KEYWORD_MAP = {normalize_token(k): k for k in SQL_KEYWORDS}


def best_keyword(token):
    best = None
    best_score = 0.0
    for kw in SQL_KEYWORDS:
        score = difflib.SequenceMatcher(None, token, kw.lower()).ratio()
        if score > best_score:
            best_score = score
            best = kw
    return best, best_score


def correct_sql_keywords(text, enable_fuzzy=True):
    if not text:
        return text

    parts = re.split(r"(\s+)", text)
    word_parts = []
    for idx, part in enumerate(parts):
        if not part or part.isspace():
            continue
        match = _WORD_RE.match(part)
        if not match:
            continue
        lead, core, trail = match.groups()
        word_parts.append([idx, lead, core, trail])

    norms = [normalize_token(item[2]) for item in word_parts]

    i = 0
    while i < len(word_parts) - 1:
        if not norms[i] or not norms[i + 1]:
            i += 1
            continue
        combined = norms[i] + norms[i + 1]
        alias = SQL_ALIAS_MAP.get(combined)
        if alias:
            word_parts[i][2] = alias
            parts[word_parts[i + 1][0]] = ""
            i += 2
            continue
        if enable_fuzzy:
            kw, score = best_keyword(combined)
            if kw and score >= MERGE_THRESHOLD:
                word_parts[i][2] = kw
                parts[word_parts[i + 1][0]] = ""
                i += 2
                continue
        i += 1

    for part_idx, lead, core, trail in word_parts:
        if parts[part_idx] == "":
            continue
        norm = normalize_token(core)
        replacement = SQL_ALIAS_MAP.get(norm) or SQL_KEYWORD_MAP.get(norm)
        if not replacement and enable_fuzzy and len(norm) >= FUZZY_MIN_LEN:
            kw, score = best_keyword(norm)
            if kw and score >= FUZZY_THRESHOLD:
                replacement = kw
        if replacement:
            parts[part_idx] = f"{lead}{replacement}{trail}"
        else:
            parts[part_idx] = f"{lead}{core}{trail}"

    return "".join(parts)
