import re
import numpy as np

def _summarize_for_log(obj, max_str_len=200):
    """Рекурсивно заменяет numpy arrays на компактное описание, остальное оставляет как есть."""
    if isinstance(obj, np.ndarray):
        return f"np.ndarray(shape={obj.shape}, dtype={obj.dtype})"
    elif isinstance(obj, (list, tuple)):
        summarized = [_summarize_for_log(item, max_str_len) for item in obj]
        return tuple(summarized) if isinstance(obj, tuple) else summarized
    elif isinstance(obj, dict):
        return {k: _summarize_for_log(v, max_str_len) for k, v in obj.items()}
    elif isinstance(obj, str):
        return obj if len(obj) <= max_str_len else obj[:max_str_len] + "..."
    else:
        return obj

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<(th|td)[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)

def _cell_text(raw: str) -> str:
    raw = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = raw.replace("|", r"\|")
    return " ".join(raw.split())

def _table_to_markdown(inner_html: str) -> str:
    rows = []
    for tr in _TR_RE.finditer(inner_html):
        cells = [_cell_text(m.group(2)) for m in _CELL_RE.finditer(tr.group(1))]
        if cells and any(c for c in cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)

def convert_html_tables(text: str) -> str:
    return _TABLE_RE.sub(lambda m: "\n\n" + _table_to_markdown(m.group(1)) + "\n\n", text)

def strip_image_tags(text: str) -> str:
    return re.sub(r"!\[.*?\]\(.*?\)\n*", "", text)
