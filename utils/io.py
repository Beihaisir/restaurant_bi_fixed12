from __future__ import annotations
import io
import re
from typing import List, Optional, Tuple

import pandas as pd

STORE_ID_RE = re.compile(r"导出人[:：]\s*(\d+)")

def _norm_cell(x) -> str:
    s = "" if x is None else str(x)
    return s.strip()

def _norm_col(c) -> str:
    # aggressive normalize: strip + remove all spaces (incl. full-width)
    s = "" if c is None else str(c)
    s = s.strip()
    s = s.replace(" ", "").replace("\u3000", "")
    return s

def _detect_store_id_from_first_rows(df_head: pd.DataFrame) -> Optional[str]:
    for r in range(min(10, len(df_head))):
        row = df_head.iloc[r].astype(str).tolist()
        for cell in row:
            m = STORE_ID_RE.search(str(cell))
            if m:
                return m.group(1)
    return None

def _find_header_row(df_head: pd.DataFrame, must_have: List[str]) -> Optional[int]:
    must = set([_norm_col(x) for x in must_have])
    for r in range(min(60, len(df_head))):
        vals = [_norm_col(x) for x in df_head.iloc[r].tolist()]
        s = set([v for v in vals if v and v != "nan"])
        if must.issubset(s):
            return r
    return None

def read_raw_noheader(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=str, encoding="utf-8", engine="python")
    return pd.read_excel(io.BytesIO(file_bytes), header=None, dtype=object)

def read_any_table(file_bytes: bytes, filename: str, must_have: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    df_raw = read_raw_noheader(file_bytes, filename)
    store_id = _detect_store_id_from_first_rows(df_raw.head(20))

    header_row = _find_header_row(df_raw.head(120), must_have)
    if header_row is None:
        header_row = 0

    df = df_raw.iloc[header_row:].copy()
    df.columns = [_norm_col(x) for x in df.iloc[0].tolist()]
    df = df.iloc[1:].reset_index(drop=True)

    # drop empty columns
    df = df.loc[:, [c for c in df.columns if c and c != "nan"]]
    return df, store_id
