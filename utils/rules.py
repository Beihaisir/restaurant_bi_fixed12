from __future__ import annotations
from dataclasses import dataclass
from typing import List

import pandas as pd

@dataclass(frozen=True)
class Rule:
    category: str
    keyword: str

def load_rules_xlsx(file_like) -> List[Rule]:
    df = pd.read_excel(file_like, sheet_name="规则表")
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "分类" not in df.columns or "关键词" not in df.columns:
        raise ValueError("分类规则模板缺少列：分类 / 关键词")
    rules: List[Rule] = []
    for _, r in df.iterrows():
        cat = str(r["分类"]).strip()
        kw = str(r["关键词"]).strip()
        if cat and kw and cat != "nan" and kw != "nan":
            rules.append(Rule(cat, kw))
    return rules

def match_categories(name: str, rules: List[Rule]) -> List[str]:
    if not isinstance(name, str):
        return []
    hits: List[str] = []
    for rule in rules:
        if rule.keyword and rule.keyword in name:
            hits.append(rule.category)
    seen = set()
    out: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out
