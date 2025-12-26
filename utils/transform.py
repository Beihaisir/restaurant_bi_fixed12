from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .rules import Rule, match_categories

JDB = "加多宝"

# 只统计这些规格（含“标准”）
SPEC_WHITELIST = ["标准", "宽面", "细面", "米饭", "宽粉", "无需主食"]


def _pick_col(df: pd.DataFrame, candidates: List[str], contains: Optional[List[str]] = None) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if contains:
        for key in contains:
            for c in cols:
                if key in str(c):
                    return c
    return None


def normalize_spec(spec_name: Any) -> Optional[str]:
    """
    只返回白名单规格（含“标准”）；其他一律返回 None（不进入规格分布）
    注意：是否统计“标准”还需要结合 类型_norm（只统计 菜品/套餐）
    """
    s = "" if spec_name is None else str(spec_name).strip()
    if not s or s == "nan":
        return None

    if "标准" in s:
        return "标准"

    if "无需主食" in s:
        return "无需主食"

    # 粉也属于宽粉
    if "宽粉" in s:
        return "宽粉"
    if "粉" in s:
        return "宽粉"

    if "米饭" in s:
        return "米饭"
    if "宽面" in s:
        return "宽面"
    if "细面" in s or "天麻面" in s:
        return "细面"

    return None


def is_standard_row(row: pd.Series) -> bool:
    spec = str(row.get("规格名称", "")).strip()
    typ = str(row.get("类型_norm", row.get("类型", ""))).strip()
    return ("标准" in spec) and (typ == "套餐")


def parse_cook_json(val: Any) -> List[Dict[str, Any]]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    s = str(val).strip()
    if not s or s == "nan" or s == "[]":
        return []
    try:
        data = json.loads(s)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def cook_add_items(cook_list: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for item in cook_list:
        name = str(item.get("name", "")).strip()
        if not name or name == JDB:
            continue
        if name.startswith("加") and len(name) >= 2:
            price = item.get("price", 0)
            try:
                p = float(price)
            except Exception:
                p = 0.0
            # 常见导出为“分”
            if p >= 10:
                p = p / 100.0
            out.append((name, p))
    return out


def dish_is_single_add(dish_name: str) -> bool:
    if not isinstance(dish_name, str):
        return False
    dish_name = dish_name.strip()
    if dish_name == JDB:
        return False
    return dish_name.startswith("加") and len(dish_name) >= 2


def dish_is_jdb(dish_name: str) -> bool:
    return isinstance(dish_name, str) and (JDB in dish_name)


def _norm_type(t: str) -> str:
    if not t or t == "nan":
        return "未知"
    if "套餐" in t:
        return "套餐"
    if "菜品" in t:
        return "菜品"
    return t


def build_fact_tables(dish_df: pd.DataFrame, pay_df: pd.DataFrame, rules: List[Rule], store_id: str):
    d = dish_df.copy()
    d["store_id"] = store_id

    # dish column mapping（兼容不同导出字段名）
    name_col = _pick_col(d, ["菜品名称"], contains=["菜品名称", "商品名称"])
    type_col = _pick_col(d, ["类型"], contains=["类型"])
    spec_col = _pick_col(d, ["规格名称"], contains=["规格"])
    cook_col = _pick_col(d, ["做法"], contains=["做法"])
    time_col = _pick_col(d, ["创建时间"], contains=["创建时间", "下单时间", "时间"])
    order_col = _pick_col(d, ["POS销售单号"], contains=["POS销售单号", "销售单号", "单号"])
    qty_col = _pick_col(d, ["菜品数量"], contains=["数量"])
    gross_col = _pick_col(d, ["小计价格"], contains=["小计价格", "小计"])
    net_col = _pick_col(d, ["优惠后小计价格"], contains=["优惠后小计价格", "优惠后小计", "实付小计", "折后小计"])
    refund_col = _pick_col(d, ["POS退款单号"], contains=["退款单号"])
    store_col = _pick_col(d, ["门店"], contains=["门店"])

    if name_col is None:
        d["菜品名称"] = None
        name_col = "菜品名称"
    if type_col is None:
        d["类型"] = ""
        type_col = "类型"
    if spec_col is None:
        d["规格名称"] = None
        spec_col = "规格名称"
    if cook_col is None:
        d["做法"] = None
        cook_col = "做法"
    if time_col is None:
        d["创建时间"] = None
        time_col = "创建时间"
    if order_col is None:
        d["POS销售单号"] = None
        order_col = "POS销售单号"
    if qty_col is None:
        d["菜品数量"] = 1
        qty_col = "菜品数量"
    if gross_col is None:
        d["小计价格"] = None
        gross_col = "小计价格"
    if net_col is None:
        d["优惠后小计价格"] = None
        net_col = "优惠后小计价格"
    if refund_col is None:
        d["POS退款单号"] = None
        refund_col = "POS退款单号"
    if store_col is None:
        d["门店"] = None
        store_col = "门店"

    d = d.rename(
        columns={
            name_col: "菜品名称",
            type_col: "类型",
            spec_col: "规格名称",
            cook_col: "做法",
            time_col: "创建时间",
            order_col: "POS销售单号",
            qty_col: "菜品数量",
            gross_col: "小计价格",
            net_col: "优惠后小计价格",
            refund_col: "POS退款单号",
            store_col: "门店",
        }
    )

    d["创建时间"] = pd.to_datetime(d["创建时间"], errors="coerce")

    # 规范化类型
    d["类型"] = d["类型"].astype(str).fillna("").str.strip()
    d["类型_norm"] = d["类型"].apply(_norm_type)

    # 规格归一（含标准）+ 标准只在 菜品/套餐 统计
    d["spec_norm"] = d["规格名称"].apply(normalize_spec)
    d.loc[(d["spec_norm"] == "标准") & (d["类型_norm"] != "套餐"), "spec_norm"] = None

    d["is_standard"] = d.apply(is_standard_row, axis=1)

    d["小计价格"] = pd.to_numeric(d["小计价格"], errors="coerce")
    d["优惠后小计价格"] = pd.to_numeric(d["优惠后小计价格"], errors="coerce")
    d["菜品数量"] = pd.to_numeric(d["菜品数量"], errors="coerce").fillna(0)

    d["dish_name"] = d["菜品名称"].astype(str)
    d["is_jdb"] = d["dish_name"].apply(dish_is_jdb)
    d["is_single_add_dish"] = d["dish_name"].apply(dish_is_single_add)

    d["categories"] = d["dish_name"].apply(lambda x: match_categories(x, rules))
    d["is_uncat"] = d["categories"].apply(lambda x: len(x) == 0)

    # main items: exclude single-add-as-dish (but keep 加多宝)
    fact_main = d[~d["is_single_add_dish"]].copy()

    # single-add fact
    add_rows: List[Dict[str, Any]] = []
    for _, row in d.iterrows():
        cook = parse_cook_json(row.get("做法"))
        for add_name, add_price in cook_add_items(cook):
            add_rows.append(
                {
                    "store_id": store_id,
                    "order_id": row.get("POS销售单号"),
                    "created_at": row.get("创建时间"),
                    "add_name": add_name,
                    "add_display": f"单加-{add_name}",
                    "qty": float(row.get("菜品数量", 1) or 1),
                    "amount": float(add_price) * float(row.get("菜品数量", 1) or 1),
                    "source": "做法",
                }
            )

    add_dish = d[d["is_single_add_dish"] & (~d["is_jdb"])].copy()
    for _, row in add_dish.iterrows():
        add_name = str(row.get("菜品名称", "")).strip()
        amt = row.get("优惠后小计价格")
        if pd.isna(amt):
            amt = row.get("小计价格")
        add_rows.append(
            {
                "store_id": store_id,
                "order_id": row.get("POS销售单号"),
                "created_at": row.get("创建时间"),
                "add_name": add_name,
                "add_display": f"单加-{add_name}",
                "qty": float(row.get("菜品数量", 1) or 1),
                "amount": float(amt or 0),
                "source": "菜品名",
            }
        )

    fact_add = pd.DataFrame(add_rows)
    if fact_add.empty:
        fact_add = pd.DataFrame(columns=["store_id", "order_id", "created_at", "add_name", "add_display", "qty", "amount", "source"])
    fact_add["created_at"] = pd.to_datetime(fact_add["created_at"], errors="coerce")

    # order-level
    fact_orders = d.groupby(["store_id", "POS销售单号"], as_index=False).agg(
        order_time=("创建时间", "min"),
        dish_lines=("菜品名称", "count"),
        dish_qty=("菜品数量", "sum"),
        gross_amount=("小计价格", "sum"),
        net_amount=("优惠后小计价格", "sum"),
        has_refund=("POS退款单号", lambda s: s.notna().any()),
    )

    # payments (robust columns)
    p = pay_df.copy()
    p["store_id"] = store_id

    order_col_p = _pick_col(p, ["POS销售单号"], contains=["POS销售单号", "销售单号", "单号"])
    paytype_col = _pick_col(p, ["支付类型"], contains=["支付类型", "支付方式", "渠道"])
    amt_col = _pick_col(p, ["总金额"], contains=["总金额", "金额", "实收", "支付金额", "支付金额(元)", "支付金额元"])

    if order_col_p is None:
        p["POS销售单号"] = None
        order_col_p = "POS销售单号"
    if paytype_col is None:
        p["支付类型"] = "未知"
        paytype_col = "支付类型"
    if amt_col is None:
        p["总金额"] = 0
        amt_col = "总金额"

    p = p.rename(columns={order_col_p: "POS销售单号", paytype_col: "支付类型", amt_col: "总金额"})
    p["总金额"] = pd.to_numeric(p["总金额"], errors="coerce").fillna(0.0)

    # attach order_time for time filter
    p = p.merge(fact_orders[["store_id", "POS销售单号", "order_time"]], on=["store_id", "POS销售单号"], how="left")

    return {
        "fact_items_main": fact_main,
        "fact_items_add": fact_add,
        "fact_pay": p,
        "fact_orders": fact_orders,
    }
