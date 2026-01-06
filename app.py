from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from utils.io import read_any_table, read_raw_noheader
from utils.rules import load_rules_xlsx, match_categories, Rule
from utils.transform import build_fact_tables

st.set_page_config(page_title="正大餐饮经营分析系统_临时", layout="wide")


@dataclass
class StoreBundle:
    store_id: str
    daily: Optional[pd.DataFrame]
    dish: Optional[pd.DataFrame]
    pay: Optional[pd.DataFrame]


DAILY_MUST = ["门店代码", "门店名称", "日期"]
DISH_MUST = ["创建时间", "菜品名称", "POS销售单号"]
PAY_MUST = ["POS销售单号", "支付类型", "总金额"]


def _as_bytes(uploaded) -> bytes:
    return uploaded.getvalue()


def _norm_colset(cols) -> set:
    out = set()
    for c in cols:
        s = str(c).strip().replace(" ", "").replace("\u3000", "")
        out.add(s)
    return out


def detect_table_kind(file_bytes: bytes, filename: str) -> Tuple[str, Optional[str]]:
    raw = read_raw_noheader(file_bytes, filename)

    store_id = None
    for r in range(min(20, len(raw))):
        for cell in raw.iloc[r].astype(str).tolist():
            if "导出人" in str(cell):
                import re

                m = re.search(r"导出人[:：]\s*(\d+)", str(cell))
                if m:
                    store_id = m.group(1)
                    break
        if store_id:
            break

    top_text = " ".join(raw.head(5).astype(str).fillna("").values.flatten().tolist())
    if "日销售报表" in top_text:
        return "daily", store_id

    df_dish, _ = read_any_table(file_bytes, filename, DISH_MUST)
    if {"POS销售单号", "菜品名称", "创建时间"}.issubset(_norm_colset(df_dish.columns)):
        return "dish", store_id

    df_pay, _ = read_any_table(file_bytes, filename, PAY_MUST)
    if {"POS销售单号", "支付类型"}.issubset(_norm_colset(df_pay.columns)):
        return "pay", store_id

    df_daily, _ = read_any_table(file_bytes, filename, DAILY_MUST)
    cols_daily = _norm_colset(df_daily.columns)
    if ("含税销售额" in cols_daily) or ("客流量" in cols_daily) or ({"门店代码", "门店名称", "日期"}.issubset(cols_daily)):
        return "daily", store_id

    return "unknown", store_id


@st.cache_data(show_spinner=False)
def parse_uploaded(files: List, rule_file) -> Tuple[List[StoreBundle], List[str], List[Rule]]:
    rules: List[Rule] = []
    if rule_file is not None:
        rules = load_rules_xlsx(io.BytesIO(_as_bytes(rule_file)))

    bundles: Dict[str, StoreBundle] = {}
    warnings: List[str] = []

    def upsert(store_id: str) -> StoreBundle:
        if store_id not in bundles:
            bundles[store_id] = StoreBundle(store_id=store_id, daily=None, dish=None, pay=None)
        return bundles[store_id]

    for f in files:
        b = _as_bytes(f)
        name = f.name
        kind, store_id = detect_table_kind(b, name)
        if store_id is None:
            store_id = "UNKNOWN"

        if kind == "daily":
            df, _ = read_any_table(b, name, DAILY_MUST)
            upsert(store_id).daily = df
        elif kind == "dish":
            df, _ = read_any_table(b, name, DISH_MUST)
            upsert(store_id).dish = df
        elif kind == "pay":
            df, _ = read_any_table(b, name, PAY_MUST)
            upsert(store_id).pay = df
        else:
            warnings.append(f"无法识别文件类型：{name}（已跳过）")

    out = list(bundles.values())
    out.sort(key=lambda x: x.store_id)
    return out, warnings, rules


def fmt_money(x: float) -> str:
    try:
        return f"¥{x:,.2f}"
    except Exception:
        return "—"

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0, None) else 0.0


def _dist_from_group(df: pd.DataFrame, key: str, value: str, topn: int = 20) -> Tuple[List[str], np.ndarray]:
    if df is None or df.empty or key not in df.columns or value not in df.columns:
        return [], np.array([], dtype=float)
    g = df.groupby(key, as_index=False)[value].sum().sort_values(value, ascending=False)
    if topn and topn > 0:
        g = g.head(topn)
    return g[key].astype(str).tolist(), g[value].astype(float).to_numpy()


def _align_two(keys_a: List[str], vals_a: np.ndarray, keys_b: List[str], vals_b: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys = list(dict.fromkeys(list(keys_a) + list(keys_b)))
    ma = {k: float(v) for k, v in zip(keys_a, vals_a)}
    mb = {k: float(v) for k, v in zip(keys_b, vals_b)}
    a = np.array([ma.get(k, 0.0) for k in keys], dtype=float)
    b = np.array([mb.get(k, 0.0) for k in keys], dtype=float)
    return keys, a, b


def _entropy_share(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    s = float(np.sum(v))
    if s <= 0:
        return 0.0
    p = v / s
    p = np.where(p <= 0, 1e-12, p)
    h = float(-np.sum(p * np.log(p)))
    hmax = float(np.log(len(p))) if len(p) > 1 else 1.0
    return h / hmax if hmax > 0 else 0.0


def _max_share(v: np.ndarray) -> float:
    s = float(np.sum(v))
    if s <= 0:
        return 0.0
    return float(np.max(v / s))


def _compute_profile_scores(filtered_store: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    o = filtered_store["orders"]
    p = filtered_store["pay"]
    m = filtered_store["items_main"]
    a = filtered_store["items_add"]

    orders = int(o["POS销售单号"].nunique()) if o is not None and not o.empty else 0
    net = float(o["net_amount"].sum()) if o is not None and not o.empty else 0.0
    paid = float(p["总金额"].sum()) if p is not None and not p.empty else 0.0
    aov = _safe_div(net, orders)
    refund_rate = float(o["has_refund"].mean()) if (o is not None and not o.empty and "has_refund" in o.columns) else 0.0
    diff_abs_rate = _safe_div(abs(net - paid), max(net, 1e-9))

    add_orders = int(a["order_id"].nunique()) if a is not None and not a.empty else 0
    add_rate = _safe_div(add_orders, orders)
    add_amt = float(a["amount"].sum()) if a is not None and not a.empty else 0.0
    add_amt_share = _safe_div(add_amt, max(net, 1e-9))

    spec_df = m[m["spec_norm"].notna()].copy() if (m is not None and not m.empty and "spec_norm" in m.columns) else pd.DataFrame()
    _, spec_vals = _dist_from_group(spec_df, "spec_norm", "菜品数量", topn=10)
    spec_entropy = _entropy_share(spec_vals) if len(spec_vals) else 0.0

    _, pay_vals = _dist_from_group(p, "支付类型", "总金额", topn=10) if (p is not None and not p.empty) else ([], np.array([], dtype=float))
    pay_entropy = _entropy_share(pay_vals) if len(pay_vals) else 0.0
    pay_max_share = _max_share(pay_vals) if len(pay_vals) else 0.0

    if m is not None and not m.empty and "categories" in m.columns:
        ex = m.copy().explode("categories")
        ex["categories"] = ex["categories"].fillna("未分类")
        cat_keys, cat_vals = _dist_from_group(ex, "categories", "优惠后小计价格", topn=50)
        if len(cat_vals):
            g = pd.DataFrame({"k": cat_keys, "v": cat_vals}).sort_values("v", ascending=False)
            top5 = float(g.head(5)["v"].sum())
            cat_top5_share = _safe_div(top5, float(g["v"].sum()))
        else:
            cat_top5_share = 0.0
    else:
        cat_top5_share = 0.0

    if o is not None and not o.empty:
        tmp = o.copy()
        tmp["slot"] = tmp["order_time"].dt.floor("30min").dt.strftime("%H:%M")
        s = tmp.groupby("slot", as_index=False).agg(v=("POS销售单号", "nunique")).sort_values("v", ascending=False)
        top3 = float(s.head(3)["v"].sum())
        peak3_share = _safe_div(top3, float(s["v"].sum()))
    else:
        peak3_share = 0.0

    return {
        "订单数": orders,
        "应收": net,
        "实收": paid,
        "客单": aov,
        "退款率": refund_rate,
        "对账差异率": diff_abs_rate,
        "单加渗透率": add_rate,
        "单加金额占比": add_amt_share,
        "规格多样性": spec_entropy,
        "渠道多样性": pay_entropy,
        "渠道最大占比": pay_max_share,
        "品类Top5占比": cat_top5_share,
        "峰值Top3占比": peak3_share,
    }


def _radar_df(scores: Dict[str, float], dims: List[str], normalize_max: Dict[str, float], invert: set) -> pd.DataFrame:
    rows = []
    for d in dims:
        v = float(scores.get(d, 0.0))
        mx = float(normalize_max.get(d, 1.0)) if normalize_max.get(d, 1.0) else 1.0
        x = v / mx if mx > 0 else 0.0
        x = max(0.0, min(1.0, x))
        if d in invert:
            x = 1.0 - x
        rows.append({"维度": d, "得分": x})
    return pd.DataFrame(rows)


def _radar_chart(df: pd.DataFrame, store_col: str = "门店") -> alt.Chart:
    """Radar chart (matplotlib-free) with strong visibility in Streamlit/Altair v6."""
    dims = df["维度"].unique().tolist()
    n = len(dims)
    if n == 0:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_point().encode(x="x", y="y")

    angle_map = {d: i * 2 * np.pi / n for i, d in enumerate(dims)}
    d2 = df.copy()
    d2["angle"] = d2["维度"].map(angle_map).astype(float)

    # Ensure score is finite
    d2["得分"] = pd.to_numeric(d2["得分"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    d2["x"] = d2["得分"] * np.cos(d2["angle"])
    d2["y"] = d2["得分"] * np.sin(d2["angle"])

    # Close polygon per store
    closed = []
    for s, g in d2.groupby(store_col):
        g = g.sort_values("angle")
        g2 = pd.concat([g, g.iloc[[0]]], ignore_index=True)
        closed.append(g2)
    d2c = pd.concat(closed, ignore_index=True)

    # Fixed domain so it never auto-zooms to zero
    enc = dict(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        color=alt.Color(f"{store_col}:N", legend=alt.Legend(title="门店")),
        tooltip=[store_col, "维度", alt.Tooltip("得分:Q", format=".2f")],
    )

    base = alt.Chart(d2c).encode(**enc)

    # Grid circles (0.25/0.5/0.75/1.0)
    rings = pd.DataFrame({"r": [0.25, 0.5, 0.75, 1.0]})
    theta = np.linspace(0, 2 * np.pi, 241)
    ring_rows = []
    for r in rings["r"]:
        for t in theta:
            ring_rows.append({"r": float(r), "x": float(r * np.cos(t)), "y": float(r * np.sin(t))})
    ring_df = pd.DataFrame(ring_rows)
    ring = alt.Chart(ring_df).mark_line(opacity=0.18, strokeWidth=2).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        detail="r:N",
    )

    # Axes lines + labels
    axes_rows = []
    for d, ang in angle_map.items():
        axes_rows.append({"维度": d, "x": 0.0, "y": 0.0, "x2": float(np.cos(ang)), "y2": float(np.sin(ang))})
    axes_df = pd.DataFrame(axes_rows)
    axes = alt.Chart(axes_df).mark_rule(opacity=0.30, strokeWidth=2).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        x2="x2:Q",
        y2="y2:Q",
    )
    labels = alt.Chart(axes_df).mark_text(align="left", dx=8, dy=8, fontSize=12).encode(
        x=alt.X("x2:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        y=alt.Y("y2:Q", axis=None, scale=alt.Scale(domain=[-1.15, 1.15])),
        text="维度:N",
    )

    # Stronger polygon + points
    poly = base.mark_line(strokeWidth=4).encode(order=alt.Order("angle:Q"))
    pts = base.mark_point(filled=True, size=140, opacity=0.95)

    return (ring + axes + poly + pts + labels).properties(height=460).configure_view(stroke=None)



import uuid

def _dl_key(tag: str, sid: str | None = None) -> str:
    """
    Generate a runtime-unique key for Streamlit elements (especially inside loops).
    Using uuid avoids DuplicateElementKey across reruns and repeated blocks.
    """
    if sid is None:
        return f"dl_{tag}_{uuid.uuid4().hex}"
    return f"dl_{tag}_{sid}_{uuid.uuid4().hex}"


def halfhour_options(min_dt: pd.Timestamp, max_dt: pd.Timestamp) -> List[pd.Timestamp]:
    if pd.isna(min_dt) or pd.isna(max_dt):
        return []
    start = min_dt.floor("30min")
    end = max_dt.ceil("30min")
    return list(pd.date_range(start, end, freq="30min"))


def apply_time_filter(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    x = df.copy()
    if col not in x.columns:
        return x
    x[col] = pd.to_datetime(x[col], errors="coerce")
    return x[(x[col] >= start) & (x[col] <= end)].copy()


def _base_items(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "类型_norm" in df.columns:
        return df[df["类型_norm"].isin(["菜品", "套餐"])].copy()
    return df[df["类型"].astype(str).str.contains("菜品|套餐", na=False)].copy()


def _share_table(df_long: pd.DataFrame, store_col: str, key_col: str, val_col: str, topn: int) -> pd.DataFrame:
    tot = df_long.groupby(key_col, as_index=False)[val_col].sum().sort_values(val_col, ascending=False).head(topn)
    keys = tot[key_col].tolist()
    sub = df_long[df_long[key_col].isin(keys)].copy()
    denom = sub.groupby(store_col, as_index=False)[val_col].sum().rename(columns={val_col: "_den"})
    sub = sub.merge(denom, on=store_col, how="left")
    sub["share"] = sub[val_col] / sub["_den"].replace(0, np.nan)
    out = sub.pivot_table(index=key_col, columns=store_col, values="share", aggfunc="sum").fillna(0.0)
    out = out.loc[keys]
    return out


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen–Shannon divergence between two non-negative vectors.
    Returns a non-negative float (0 means identical distributions).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Handle empty/zero vectors robustly
    ps = float(np.nansum(p))
    qs = float(np.nansum(q))
    if ps <= 0 and qs <= 0:
        return 0.0
    if ps <= 0:
        p = np.zeros_like(q, dtype=float)
        ps = 1.0
    if qs <= 0:
        q = np.zeros_like(p, dtype=float)
        qs = 1.0

    p = np.where(np.isfinite(p), p, 0.0) / ps
    q = np.where(np.isfinite(q), q, 0.0) / qs
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        x = np.where(x <= 0, 1e-12, x)
        y = np.where(y <= 0, 1e-12, y)
        return float(np.sum(x * np.log(x / y)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def _simple_gradient_styler(df: pd.DataFrame):
    """
    Matplotlib-free gradient styler for Streamlit Cloud.
    pandas Styler.background_gradient requires matplotlib; this does not.
    """
    if df is None or df.empty:
        return df

    vals = df.to_numpy(dtype=float, copy=True)
    finite = np.isfinite(vals)
    if not finite.any():
        return df

    vmin = float(np.nanmin(vals[finite]))
    vmax = float(np.nanmax(vals[finite]))
    if vmin == vmax:
        vmax = vmin + 1.0

    def _color(v):
        if v is None:
            return ""
        try:
            fv = float(v)
        except Exception:
            return ""
        if np.isnan(fv) or np.isinf(fv):
            return ""
        x = (fv - vmin) / (vmax - vmin)
        x = max(0.0, min(1.0, x))
        # light -> dark blue
        r1, g1, b1 = (246, 248, 255)
        r2, g2, b2 = (30, 98, 211)
        r = int(r1 + (r2 - r1) * x)
        g = int(g1 + (g2 - g1) * x)
        b = int(b1 + (b2 - b1) * x)
        txt = "#ffffff" if x > 0.55 else "#111111"
        return f"background-color: rgb({r},{g},{b}); color: {txt};"

    return df.style.applymap(_color)

    def _kl(x, y):
        x = np.where(x <= 0, 1e-12, x)
        y = np.where(y <= 0, 1e-12, y)
        return float(np.sum(x * np.log(x / y)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def main() -> None:
    st.title("🍽️ 正大餐饮经营分析系统_临时版")

    with st.sidebar:
        st.header("数据输入")
        rule_file = st.file_uploader("上传：分类规则模板（xlsx，Sheet=规则表）", type=["xlsx"], accept_multiple_files=False)
        files = st.file_uploader(
            "上传：三类报表（可多门店、多文件；支持 xls/xlsx/csv）",
            type=["xls", "xlsx", "csv"],
            accept_multiple_files=True,
        )
        st.caption("口径：时间最小30分钟；“加xx”为单加（加多宝除外）；天麻面并入细面；“标准”仅统计为【套餐】的标准行。")

    if not files:
        st.info("请先在左侧上传报表文件。")
        return

    bundles, warnings, rules = parse_uploaded(files, rule_file)

    if warnings:
        with st.expander("⚠️ 文件识别警告", expanded=False):
            for w in warnings:
                st.warning(w)

    analyzable = [b for b in bundles if b.dish is not None and b.pay is not None and b.daily is not None and b.store_id != "UNKNOWN"]
    missing = [b for b in bundles if b not in analyzable]

    if missing:
        with st.expander("⚠️ 缺表门店（不进入分析）", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [{"store_id": b.store_id, "有日销售": b.daily is not None, "有菜品明细": b.dish is not None, "有支付明细": b.pay is not None} for b in missing]
                ),
                use_container_width=True,
            )

    if not analyzable:
        st.error("没有“三表齐全”的门店，无法分析。")
        return

    store_ids = [b.store_id for b in analyzable]
    sel_stores = st.multiselect("选择门店（支持多店对比）", options=store_ids, default=store_ids[:1])
    if not sel_stores:
        st.stop()

    facts_by_store: Dict[str, Dict[str, pd.DataFrame]] = {}
    for b in analyzable:
        if b.store_id in sel_stores:
            facts_by_store[b.store_id] = build_fact_tables(b.dish, b.pay, rules, b.store_id)

    all_orders = pd.concat([facts_by_store[s]["fact_orders"] for s in sel_stores], ignore_index=True)
    min_dt = all_orders["order_time"].min()
    max_dt = all_orders["order_time"].max()
    opts = halfhour_options(min_dt, max_dt)
    if not opts:
        st.error("无法从数据中解析创建时间。")
        return

    c1, c2 = st.columns(2)
    with c1:
        start = st.selectbox("开始时间（30分钟粒度）", options=opts, index=0, format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))
    with c2:
        end = st.selectbox("结束时间（30分钟粒度）", options=opts, index=len(opts) - 1, format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))

    if start > end:
        st.error("开始时间不能晚于结束时间。")
        return

    filtered: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sid in sel_stores:
        f = facts_by_store[sid]
        filtered[sid] = {
            "items_main": apply_time_filter(f["fact_items_main"], "创建时间", start, end),
            "items_add": apply_time_filter(f["fact_items_add"], "created_at", start, end),
            "pay": apply_time_filter(f["fact_pay"], "order_time", start, end),
            "orders": apply_time_filter(f["fact_orders"], "order_time", start, end),
        }

    tabs = st.tabs(
        [
            "① 董事/股东总览",
            "② 门店对比",
            "③ 规格",
            "④ 品类结构",
            "⑤ 单加分析",
            "⑥ 支付渠道",
            "⑦ 退款/异常与对账",
            "⑧ 未分类池（可导出）",
            "⑨ 明细导出",
            "⑩ 时段热力图",
            "⑪ 门店画像卡（两店对比）",
        ]
    )

    # ① 总览
    with tabs[0]:
        st.subheader("董事/股东视角：规模、效率、结构、风险")

        rows = []
        for sid in sel_stores:
            o = filtered[sid]["orders"]
            p = filtered[sid]["pay"]
            orders = int(o["POS销售单号"].nunique()) if not o.empty else 0
            rows.append(
                {
                    "store_id": sid,
                    "订单数": orders,
                    "菜品销量": float(o["dish_qty"].sum()) if not o.empty else 0.0,
                    "菜品应收(优惠后)": float(o["net_amount"].sum()) if not o.empty else 0.0,
                    "支付实收": float(p["总金额"].sum()) if not p.empty else 0.0,
                    "退款单占比": float(o["has_refund"].mean()) if not o.empty else 0.0,
                    "客单(应收/订单)": (float(o["net_amount"].sum()) / orders) if orders else np.nan,
                }
            )
        dfk = pd.DataFrame(rows)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("选中门店订单数", int(dfk["订单数"].sum()))
        k2.metric("选中门店菜品销量", f"{dfk['菜品销量'].sum():,.0f}")
        k3.metric("选中门店菜品应收(优惠后)", fmt_money(dfk["菜品应收(优惠后)"].sum()))
        k4.metric("选中门店支付实收", fmt_money(dfk["支付实收"].sum()))
        st.dataframe(dfk, use_container_width=True)

        oall = pd.concat([filtered[s]["orders"] for s in sel_stores], ignore_index=True)
        if not oall.empty:
            oall["bucket"] = oall["order_time"].dt.floor("30min")
            trend = oall.groupby("bucket", as_index=False).agg(订单数=("POS销售单号", "nunique"), 菜品应收=("net_amount", "sum")).sort_values("bucket")
            st.line_chart(trend.set_index("bucket")[["订单数", "菜品应收"]])
            st.markdown("**峰值时段 Top10（按订单数）**")
            st.dataframe(trend.sort_values("订单数", ascending=False).head(10), use_container_width=True)

        main_all = pd.concat([filtered[s]["items_main"] for s in sel_stores], ignore_index=True)
        base_items = _base_items(main_all)
        if not base_items.empty:
            top_rev = (
                base_items.groupby("菜品名称", as_index=False)
                .agg(应收=("优惠后小计价格", "sum"), 销量=("菜品数量", "sum"), 订单数=("POS销售单号", "nunique"))
                .sort_values(["应收", "销量"], ascending=False)
                .head(20)
            )
            st.markdown("### Top20 菜品（按应收排序）")
            st.dataframe(top_rev, use_container_width=True)
            st.bar_chart(top_rev.set_index("菜品名称")[["应收"]])

            dish_rev = base_items.groupby("菜品名称", as_index=False).agg(应收=("优惠后小计价格", "sum")).sort_values("应收", ascending=False)
            dish_rev["累计应收"] = dish_rev["应收"].cumsum()
            total_rev = dish_rev["应收"].sum()
            dish_rev["累计占比"] = dish_rev["累计应收"] / total_rev if total_rev else 0
            n80 = int((dish_rev["累计占比"] <= 0.8).sum() + 1) if total_rev else 0
            st.markdown("### 爆品/长尾（帕累托）")
            st.write(f"达到 **80%应收** 需要的菜品数：**{n80}** / 总菜品数 {len(dish_rev)}")
            st.dataframe(dish_rev.head(50), use_container_width=True)

        add_all = pd.concat([filtered[s]["items_add"] for s in sel_stores], ignore_index=True)
        if not add_all.empty:
            top_add = add_all.groupby("add_display", as_index=False).agg(单加金额=("amount", "sum"), 销量=("qty", "sum"), 订单数=("order_id", "nunique")).sort_values(["单加金额", "销量"], ascending=False).head(20)
            st.markdown("### Top20 单加（按单加金额排序）")
            st.dataframe(top_add, use_container_width=True)
            st.bar_chart(top_add.set_index("add_display")[["单加金额"]])

    # ② 门店对比（结构对比 + 偏离度：修复sel_stores作用域）
    with tabs[1]:
        st.subheader("门店对比：同口径看差异（店长/区域经理/总部）")
        rows = []
        for sid in sel_stores:
            o = filtered[sid]["orders"]
            p = filtered[sid]["pay"]
            orders = int(o["POS销售单号"].nunique()) if not o.empty else 0
            net = float(o["net_amount"].sum()) if not o.empty else 0.0
            paid = float(p["总金额"].sum()) if not p.empty else 0.0
            rows.append({"store_id": sid, "订单数": orders, "应收(优惠后)": net, "实收": paid, "应收-实收差异": net - paid, "客单(应收/订单)": (net / orders) if orders else np.nan})
        df = pd.DataFrame(rows).sort_values("应收(优惠后)", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("store_id")[["应收(优惠后)", "实收"]])
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(df.set_index("store_id")[["应收-实收差异"]])
        with c2:
            st.bar_chart(df.set_index("store_id")[["客单(应收/订单)"]])

        st.markdown("### 同店结构对比（总部/区域：找偏离、找可复制打法）")
        dim = st.selectbox("选择结构维度", options=["规格结构", "品类结构", "单加结构", "支付结构"], index=0)
        metric = st.selectbox("选择指标", options=["应收", "销量/笔数"], index=0, key="cmp_metric")

        def build_long() -> pd.DataFrame:
            rows2 = []
            if dim == "规格结构":
                for sid in sel_stores:
                    m = _base_items(filtered[sid]["items_main"])
                    x = m[m["spec_norm"].notna()].copy() if not m.empty else pd.DataFrame()
                    if x.empty:
                        continue
                    if metric == "应收":
                        g = x.groupby("spec_norm", as_index=False).agg(v=("优惠后小计价格", "sum"))
                    else:
                        g = x.groupby("spec_norm", as_index=False).agg(v=("菜品数量", "sum"))
                    g["store_id"] = sid
                    g = g.rename(columns={"spec_norm": "k"})
                    rows2.append(g)
            elif dim == "品类结构":
                for sid in sel_stores:
                    m = filtered[sid]["items_main"]
                    if m.empty:
                        continue
                    ex = m.copy().explode("categories")
                    ex["categories"] = ex["categories"].fillna("未分类")
                    if metric == "应收":
                        g = ex.groupby("categories", as_index=False).agg(v=("优惠后小计价格", "sum"))
                    else:
                        g = ex.groupby("categories", as_index=False).agg(v=("菜品数量", "sum"))
                    g["store_id"] = sid
                    g = g.rename(columns={"categories": "k"})
                    rows2.append(g)
            elif dim == "单加结构":
                for sid in sel_stores:
                    a = filtered[sid]["items_add"]
                    if a.empty:
                        continue
                    if metric == "应收":
                        g = a.groupby("add_display", as_index=False).agg(v=("amount", "sum"))
                    else:
                        g = a.groupby("add_display", as_index=False).agg(v=("order_id", "nunique"))
                    g["store_id"] = sid
                    g = g.rename(columns={"add_display": "k"})
                    rows2.append(g)
            else:
                for sid in sel_stores:
                    p = filtered[sid]["pay"]
                    if p.empty:
                        continue
                    if metric == "应收":
                        g = p.groupby("支付类型", as_index=False).agg(v=("总金额", "sum"))
                    else:
                        g = p.groupby("支付类型", as_index=False).agg(v=("POS销售单号", "count"))
                    g["store_id"] = sid
                    g = g.rename(columns={"支付类型": "k"})
                    rows2.append(g)

            if rows2:
                return pd.concat(rows2, ignore_index=True)
            return pd.DataFrame(columns=["store_id", "k", "v"])

        long = build_long()
        if long.empty:
            st.info("暂无数据用于结构对比。")
        else:
            topn = 6 if dim == "规格结构" else (12 if dim == "品类结构" else 10)
            share = _share_table(long, "store_id", "k", "v", topn=topn)
            st.dataframe(share.style.format("{:.1%}"), use_container_width=True)

            chart = alt.Chart(long).mark_bar().encode(
                x=alt.X("store_id:N", title="门店"),
                y=alt.Y("v:Q", title="值"),
                color=alt.Color("k:N", title=dim.replace("结构", "")),
                tooltip=["store_id", "k", "v"],
            ).properties(height=420)
            st.altair_chart(chart, use_container_width=True)

            # 偏离度
            st.markdown("### 偏离度排名：哪家门店最‘不一样’？哪家可做标杆？")
            mat = share.T  # store x key
            mean = mat.mean(axis=0).values
            rows3 = []
            for sid in mat.index:
                js = _js_divergence(mat.loc[sid].values, mean)
                rows3.append({"store_id": sid, "偏离度(JS)": js})
            ddf = pd.DataFrame(rows3).sort_values("偏离度(JS)", ascending=False)
            st.dataframe(ddf, use_container_width=True)
            if len(ddf) >= 2:
                bench = ddf.sort_values("偏离度(JS)", ascending=True).iloc[0]["store_id"]
                outlier = ddf.iloc[0]["store_id"]
                st.write(f"**建议**：可先把偏离度最低的门店 **{bench}** 作为“标杆结构”，重点复盘偏离度最高的门店 **{outlier}** 的原因（客群/时段/套餐占比/渠道）。")

    # ③ 规格
    with tabs[2]:
        st.subheader("规格：主食结构（含“标准”=套餐标准）")
        st.caption("规格分布只统计：标准 / 宽面 / 细面(含天麻面) / 米饭 / 宽粉(含粉) / 无需主食；“标准”仅来源于 类型=套餐 的标准行。")
        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            if m.empty:
                st.info("无数据")
                continue
            base = _base_items(m)
            spec_base = base[base["spec_norm"].notna()].copy()
            if spec_base.empty:
                st.info("该时间范围内没有命中规格白名单的数据。")
                continue
            spec = spec_base.groupby("spec_norm", as_index=False).agg(销量=("菜品数量", "sum"), 应收=("优惠后小计价格", "sum"), 行数=("菜品名称", "count"), 订单数=("POS销售单号", "nunique")).sort_values(["销量", "应收"], ascending=False)
            spec["销量占比"] = spec["销量"] / spec["销量"].sum() if spec["销量"].sum() else 0
            spec["应收占比"] = spec["应收"] / spec["应收"].sum() if spec["应收"].sum() else 0
            st.dataframe(spec, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(spec.set_index("spec_norm")[["销量"]])
            with c2:
                st.bar_chart(spec.set_index("spec_norm")[["应收"]])
            spec_base["bucket"] = spec_base["创建时间"].dt.floor("30min")
            top_specs = spec["spec_norm"].head(5).tolist()
            pivot = spec_base[spec_base["spec_norm"].isin(top_specs)].groupby(["bucket", "spec_norm"], as_index=False).agg(销量=("菜品数量", "sum"))
            if not pivot.empty:
                piv = pivot.pivot(index="bucket", columns="spec_norm", values="销量").fillna(0).sort_index()
                st.line_chart(piv)

    # ④ 品类结构
    with tabs[3]:
        st.subheader("品类结构：规则模板命中（多标签）")
        st.caption("一个菜品可命中多个分类，命中即各计一次；未命中进入未分类池。")
        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            if m.empty:
                st.info("无数据")
                continue
            exploded = m.copy().explode("categories")
            exploded["categories"] = exploded["categories"].fillna("未分类")
            cat = exploded.groupby("categories", as_index=False).agg(销量=("菜品数量", "sum"), 应收=("优惠后小计价格", "sum"), 菜品行数=("菜品名称", "count")).sort_values("应收", ascending=False)
            st.dataframe(cat, use_container_width=True)
            st.bar_chart(cat.set_index("categories")[["应收"]])
            topn = st.slider(f"TopN 菜品（门店 {sid}）", min_value=5, max_value=50, value=20, step=5, key=f"topn_{sid}")
            cats = ["全部"] + sorted(exploded["categories"].dropna().unique().tolist())
            sel_cat = st.selectbox(f"选择分类（门店 {sid}）", options=cats, key=f"selcat_{sid}")
            view = exploded if sel_cat == "全部" else exploded[exploded["categories"] == sel_cat]
            top_items = view.groupby("菜品名称", as_index=False).agg(应收=("优惠后小计价格", "sum"), 销量=("菜品数量", "sum"), 订单数=("POS销售单号", "nunique")).sort_values(["应收", "销量"], ascending=False).head(topn)
            st.dataframe(top_items, use_container_width=True)

    # ⑤ 单加分析
    with tabs[4]:
        st.subheader("单加分析：加料带来的结构与客单提升（与主菜严格隔离）")
        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            a = filtered[sid]["items_add"]
            if a.empty:
                st.info("无单加记录")
                continue
            add = a.groupby("add_display", as_index=False).agg(销量=("qty", "sum"), 单加金额=("amount", "sum"), 订单数=("order_id", "nunique"), 来源=("source", lambda s: ",".join(sorted(set(map(str, s)))))).sort_values(["单加金额", "销量"], ascending=False)
            st.dataframe(add, use_container_width=True)
            st.bar_chart(add.set_index("add_display")[["单加金额"]])
            orders = filtered[sid]["orders"]
            add_orders = int(a["order_id"].nunique())
            total_orders = int(orders["POS销售单号"].nunique()) if not orders.empty else 0
            st.metric("单加渗透率（含单加订单/总订单）", f"{(add_orders / total_orders * 100) if total_orders else 0:.1f}%")
            if not orders.empty:
                add_set = set(a["order_id"].dropna().astype(str).tolist())
                o2 = orders.copy()
                o2["has_add"] = o2["POS销售单号"].astype(str).isin(add_set)
                grp = o2.groupby("has_add", as_index=False).agg(订单数=("POS销售单号", "nunique"), 应收=("net_amount", "sum"))
                grp["客单(应收/订单)"] = grp["应收"] / grp["订单数"].replace(0, np.nan)
                st.markdown("**有单加 vs 无单加（客单提升）**")
                st.dataframe(grp, use_container_width=True)

    # ⑥ 支付渠道
    with tabs[5]:
        st.subheader("支付渠道：渠道结构、团购渗透、混合支付")
        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            p = filtered[sid]["pay"]
            if p.empty:
                st.warning("无支付数据（该门店在筛选时间范围内支付表未关联到任何订单，或支付表未被正确识别）")
                continue
            pay = p.groupby("支付类型", as_index=False).agg(实收=("总金额", "sum"), 支付笔数=("POS销售单号", "count"), 涉及订单=("POS销售单号", "nunique")).sort_values(["实收", "支付笔数"], ascending=False)
            st.dataframe(pay, use_container_width=True)

    # ⑦ 退款/异常与对账（保持fixed10功能 + 增强已在上面实现）
    

        if p is not None and not p.empty:
            pay_kind = p.groupby(["store_id", "POS销售单号"], as_index=False).agg(
                paid=("总金额", "sum"),
                k=("支付类型", lambda s: "混合" if s.nunique() > 1 else str(list(s)[0])),
            )
            rr = o.merge(pay_kind, on=["store_id", "POS销售单号"], how="left")
            rr["paid"] = rr["paid"].fillna(0.0)
            rr["k"] = rr["k"].fillna("未知")
            rr["diff"] = rr["net_amount"] - rr["paid"]

            byk = rr.groupby("k", as_index=False).agg(
                订单数=("POS销售单号", "nunique"),
                应收=("net_amount", "sum"),
                实收=("paid", "sum"),
                差异=("diff", "sum"),
            ).sort_values(["差异", "订单数"], ascending=False)
            st.markdown("**按支付渠道分解（差异=应收-实收）**")
            st.dataframe(byk, use_container_width=True)
            st.bar_chart(byk.set_index("k")[["差异"]])

            rr["slot"] = rr["order_time"].dt.floor("30min")
            bys = rr.groupby("slot", as_index=False).agg(
                订单数=("POS销售单号", "nunique"),
                应收=("net_amount", "sum"),
                实收=("paid", "sum"),
                差异=("diff", "sum"),
            ).sort_values("slot")
            st.markdown("**按半小时分解（差异趋势）**")
            st.line_chart(bys.set_index("slot")[["差异", "应收", "实收"]])

        refund_rate = float(o["has_refund"].mean()) if ("has_refund" in o.columns and not o.empty) else 0.0
        st.metric("退款单占比（菜品表存在POS退款单号）", f"{refund_rate * 100:.1f}%")

        st.markdown("### 退款归因：菜品/时段/渠道/规格/品类（可导出）")
        items = filtered[sid]["items_main"]
        ref_rows = items[items["POS退款单号"].notna()].copy() if (items is not None and not items.empty and "POS退款单号" in items.columns) else pd.DataFrame()
        if ref_rows.empty:
            st.info("该时间范围内未识别到退款行（POS退款单号 为空）。")
        else:
            ref_rows["slot"] = ref_rows["创建时间"].dt.floor("30min")

            top_dish = ref_rows.groupby("菜品名称", as_index=False).agg(
                退款行数=("菜品名称", "count"),
                退款应收=("优惠后小计价格", "sum"),
                退款订单=("POS销售单号", "nunique"),
            ).sort_values(["退款应收", "退款行数"], ascending=False).head(30)
            st.markdown("**退款Top菜品（按退款应收）**")
            st.dataframe(top_dish, use_container_width=True)
            st.bar_chart(top_dish.set_index("菜品名称")[["退款应收"]])

            top_slot = ref_rows.groupby("slot", as_index=False).agg(
                退款应收=("优惠后小计价格", "sum"),
                退款订单=("POS销售单号", "nunique"),
            ).sort_values("slot")
            st.markdown("**退款时段趋势（半小时）**")
            st.line_chart(top_slot.set_index("slot")[["退款应收", "退款订单"]])

            if "spec_norm" in ref_rows.columns:
                top_spec = ref_rows.groupby("spec_norm", as_index=False).agg(
                    退款应收=("优惠后小计价格", "sum"),
                    退款行数=("菜品名称", "count"),
                ).sort_values("退款应收", ascending=False)
                st.markdown("**退款按规格**")
                st.dataframe(top_spec, use_container_width=True)

            if "categories" in ref_rows.columns:
                ex = ref_rows.copy().explode("categories")
                ex["categories"] = ex["categories"].fillna("未分类")
                top_cat = ex.groupby("categories", as_index=False).agg(
                    退款应收=("优惠后小计价格", "sum"),
                    退款行数=("菜品名称", "count"),
                ).sort_values("退款应收", ascending=False).head(30)
                st.markdown("**退款按品类**")
                st.dataframe(top_cat, use_container_width=True)

            if p is not None and not p.empty:
                pay_kind2 = p.groupby(["store_id", "POS销售单号"], as_index=False).agg(
                    k=("支付类型", lambda s: "混合" if s.nunique() > 1 else str(list(s)[0]))
                )
                ref_o = ref_rows[["store_id", "POS销售单号"]].drop_duplicates().merge(
                    pay_kind2, on=["store_id", "POS销售单号"], how="left"
                )
                ref_o["k"] = ref_o["k"].fillna("未知")
                top_k = ref_o.groupby("k", as_index=False).agg(退款订单=("POS销售单号", "nunique")).sort_values("退款订单", ascending=False)
                st.markdown("**退款订单按支付渠道（订单口径）**")
                st.dataframe(top_k, use_container_width=True)

            add_orders = set(filtered[sid]["items_add"]["order_id"].dropna().astype(str).tolist()) if (filtered[sid]["items_add"] is not None and not filtered[sid]["items_add"].empty) else set()
            refund_orders = set(ref_rows["POS销售单号"].dropna().astype(str).unique().tolist())
            has_add_rate = (len(refund_orders & add_orders) / len(refund_orders)) if refund_orders else 0.0
            st.metric("退款订单含单加占比", f"{has_add_rate*100:.1f}%")

            if add_orders and not filtered[sid]["items_add"].empty:
                a_ref = filtered[sid]["items_add"].copy()
                a_ref = a_ref[a_ref["order_id"].astype(str).isin(refund_orders)]
                if not a_ref.empty:
                    top_add_ref = a_ref.groupby("add_display", as_index=False).agg(
                        单加金额=("amount", "sum"),
                        单加订单=("order_id", "nunique"),
                    ).sort_values("单加金额", ascending=False).head(20)
                    st.markdown("**退款订单中的单加Top（按单加金额）**")
                    st.dataframe(top_add_ref, use_container_width=True)

            st.download_button(
                f"导出退款明细（{sid}）CSV",
                key=_dl_key("ln696", sid) ,
                data=ref_rows.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"退款明细_{sid}.csv",
                mime="text/csv",
            )



    # ⑦ 退款/异常与对账
    with tabs[6]:
        st.subheader("退款/异常与对账：抓风险、抓漏损、抓口径问题")
        st.caption("对账口径：菜品应收（优惠后小计求和） vs 支付实收（支付表金额求和）。支持按渠道/半小时拆分差异，并做退款归因。")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            o = filtered[sid]["orders"]
            p = filtered[sid]["pay"]
            if o.empty:
                st.info("无数据")
                continue

            paid_by_order = (
                p.groupby(["store_id", "POS销售单号"], as_index=False).agg(paid=("总金额", "sum"))
                if (p is not None and not p.empty)
                else pd.DataFrame(columns=["store_id", "POS销售单号", "paid"])
            )
            r = o.merge(paid_by_order, on=["store_id", "POS销售单号"], how="left")
            r["paid"] = r["paid"].fillna(0.0)
            r["diff"] = r["net_amount"] - r["paid"]

            c1, c2, c3 = st.columns(3)
            c1.metric("订单数", int(r["POS销售单号"].nunique()))
            c2.metric("应收(优惠后)", fmt_money(float(r["net_amount"].sum())))
            c3.metric("实收", fmt_money(float(r["paid"].sum())))

            st.markdown("**差异Top订单（应收-实收）**")
            top_diff = r.sort_values("diff", ascending=False).head(200)
            st.dataframe(top_diff, use_container_width=True)
            st.download_button(
                f"导出差异Top订单（{sid}）CSV",
                key=_dl_key("ln736", sid) ,
                data=top_diff.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"对账差异Top_{sid}.csv",
                mime="text/csv",
            )

            if p is not None and not p.empty:
                pay_kind = p.groupby(["store_id", "POS销售单号"], as_index=False).agg(
                    paid=("总金额", "sum"),
                    k=("支付类型", lambda s: "混合" if s.nunique() > 1 else str(list(s)[0])),
                )
                rr = o.merge(pay_kind, on=["store_id", "POS销售单号"], how="left")
                rr["paid"] = rr["paid"].fillna(0.0)
                rr["k"] = rr["k"].fillna("未知")
                rr["diff"] = rr["net_amount"] - rr["paid"]

                byk = rr.groupby("k", as_index=False).agg(
                    订单数=("POS销售单号", "nunique"),
                    应收=("net_amount", "sum"),
                    实收=("paid", "sum"),
                    差异=("diff", "sum"),
                ).sort_values(["差异", "订单数"], ascending=False)
                st.markdown("**按支付渠道分解（差异=应收-实收）**")
                st.dataframe(byk, use_container_width=True)
                st.bar_chart(byk.set_index("k")[["差异"]])

                rr["slot"] = rr["order_time"].dt.floor("30min")
                bys = rr.groupby("slot", as_index=False).agg(
                    订单数=("POS销售单号", "nunique"),
                    应收=("net_amount", "sum"),
                    实收=("paid", "sum"),
                    差异=("diff", "sum"),
                ).sort_values("slot")
                st.markdown("**按半小时分解（差异趋势）**")
                st.line_chart(bys.set_index("slot")[["差异", "应收", "实收"]])

            refund_rate = float(o["has_refund"].mean()) if ("has_refund" in o.columns and not o.empty) else 0.0
            st.metric("退款单占比（菜品表存在POS退款单号）", f"{refund_rate * 100:.1f}%")

            st.markdown("### 退款归因：菜品/时段/渠道/规格/品类（可导出）")
            items = filtered[sid]["items_main"]
            ref_rows = (
                items[items["POS退款单号"].notna()].copy()
                if (items is not None and not items.empty and "POS退款单号" in items.columns)
                else pd.DataFrame()
            )
            if ref_rows.empty:
                st.info("该时间范围内未识别到退款行（POS退款单号 为空）。")
                continue

            ref_rows["slot"] = ref_rows["创建时间"].dt.floor("30min")

            top_dish = ref_rows.groupby("菜品名称", as_index=False).agg(
                退款行数=("菜品名称", "count"),
                退款应收=("优惠后小计价格", "sum"),
                退款订单=("POS销售单号", "nunique"),
            ).sort_values(["退款应收", "退款行数"], ascending=False).head(30)
            st.markdown("**退款Top菜品（按退款应收）**")
            st.dataframe(top_dish, use_container_width=True)
            st.bar_chart(top_dish.set_index("菜品名称")[["退款应收"]])

            top_slot = ref_rows.groupby("slot", as_index=False).agg(
                退款应收=("优惠后小计价格", "sum"),
                退款订单=("POS销售单号", "nunique"),
            ).sort_values("slot")
            st.markdown("**退款时段趋势（半小时）**")
            st.line_chart(top_slot.set_index("slot")[["退款应收", "退款订单"]])

            if "spec_norm" in ref_rows.columns:
                top_spec = ref_rows.groupby("spec_norm", as_index=False).agg(
                    退款应收=("优惠后小计价格", "sum"),
                    退款行数=("菜品名称", "count"),
                ).sort_values("退款应收", ascending=False)
                st.markdown("**退款按规格**")
                st.dataframe(top_spec, use_container_width=True)

            if "categories" in ref_rows.columns:
                ex = ref_rows.copy().explode("categories")
                ex["categories"] = ex["categories"].fillna("未分类")
                top_cat = ex.groupby("categories", as_index=False).agg(
                    退款应收=("优惠后小计价格", "sum"),
                    退款行数=("菜品名称", "count"),
                ).sort_values("退款应收", ascending=False).head(30)
                st.markdown("**退款按品类**")
                st.dataframe(top_cat, use_container_width=True)

            if p is not None and not p.empty:
                pay_kind2 = p.groupby(["store_id", "POS销售单号"], as_index=False).agg(
                    k=("支付类型", lambda s: "混合" if s.nunique() > 1 else str(list(s)[0]))
                )
                ref_o = ref_rows[["store_id", "POS销售单号"]].drop_duplicates().merge(
                    pay_kind2, on=["store_id", "POS销售单号"], how="left"
                )
                ref_o["k"] = ref_o["k"].fillna("未知")
                top_k = ref_o.groupby("k", as_index=False).agg(退款订单=("POS销售单号", "nunique")).sort_values("退款订单", ascending=False)
                st.markdown("**退款订单按支付渠道（订单口径）**")
                st.dataframe(top_k, use_container_width=True)

            add_orders = set(filtered[sid]["items_add"]["order_id"].dropna().astype(str).tolist()) if (filtered[sid]["items_add"] is not None and not filtered[sid]["items_add"].empty) else set()
            refund_orders = set(ref_rows["POS销售单号"].dropna().astype(str).unique().tolist())
            has_add_rate = (len(refund_orders & add_orders) / len(refund_orders)) if refund_orders else 0.0
            st.metric("退款订单含单加占比", f"{has_add_rate*100:.1f}%")

            if add_orders and not filtered[sid]["items_add"].empty:
                a_ref = filtered[sid]["items_add"].copy()
                a_ref = a_ref[a_ref["order_id"].astype(str).isin(refund_orders)]
                if not a_ref.empty:
                    top_add_ref = a_ref.groupby("add_display", as_index=False).agg(
                        单加金额=("amount", "sum"),
                        单加订单=("order_id", "nunique"),
                    ).sort_values("单加金额", ascending=False).head(20)
                    st.markdown("**退款订单中的单加Top（按单加金额）**")
                    st.dataframe(top_add_ref, use_container_width=True)

            st.download_button(
                f"导出退款明细（{sid}）CSV",
                key=_dl_key("ln852", sid) ,
                data=ref_rows.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"退款明细_{sid}.csv",
                mime="text/csv",
            )

    # ⑧ 未分类池
    with tabs[7]:
        st.subheader("未分类池：可查看、可导出（规则迭代入口）")
        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            if m.empty:
                st.info("无数据")
                continue
            un = m[m["categories"].apply(lambda x: len(x) == 0)].copy()
            st.write(f"未分类主菜行数：{len(un):,}")
            st.dataframe(un.head(200), use_container_width=True)
            key=_dl_key("ln872", sid) ,
            st.download_button(f"导出未分类主菜（{sid}）CSV", data=un.to_csv(index=False).encode("utf-8-sig"), file_name=f"未分类主菜_{sid}.csv", mime="text/csv")

    # ⑨ 明细导出
    with tabs[8]:
        st.subheader("明细导出：总部/财务/店长二次分析")
        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            key=_dl_key("ln880", sid) ,
            st.download_button(f"导出菜品明细-过滤后（{sid}）CSV", data=m.to_csv(index=False).encode("utf-8-sig"), file_name=f"菜品明细_过滤后_{sid}.csv", mime="text/csv")

    # ⑩ 时段热力图（动作建议+导出）
    with tabs[9]:
        st.subheader("时段热力图（30分钟粒度）：峰谷、排班、备货、渠道动作")
        st.caption("行=日期，列=半小时；可选择指标；支持选中门店汇总或按门店分别查看。")

        metric = st.selectbox("选择指标", options=["订单数", "应收(优惠后)", "实收", "客单(应收/订单)", "单加渗透率(含单加订单/总订单)"], index=0)
        scope = st.radio("范围", options=["选中门店汇总", "按门店分别看"], horizontal=True)

        def _build_heat(o_df: pd.DataFrame, p_df: pd.DataFrame, a_df: pd.DataFrame) -> Optional[pd.DataFrame]:
            if o_df is None or o_df.empty:
                return None
            o = o_df.copy()
            o["date"] = o["order_time"].dt.date
            o["slot"] = o["order_time"].dt.floor("30min").dt.strftime("%H:%M")
            grp = o.groupby(["date", "slot"], as_index=False).agg(orders=("POS销售单号", "nunique"), net=("net_amount", "sum"))
            if p_df is not None and not p_df.empty:
                p = p_df.copy()
                p["date"] = p["order_time"].dt.date
                p["slot"] = p["order_time"].dt.floor("30min").dt.strftime("%H:%M")
                paid = p.groupby(["date", "slot"], as_index=False).agg(paid=("总金额", "sum"))
                grp = grp.merge(paid, on=["date", "slot"], how="left")
            grp["paid"] = grp.get("paid", 0).fillna(0.0)
            if a_df is not None and not a_df.empty:
                a = a_df.copy()
                a["date"] = a["created_at"].dt.date
                a["slot"] = a["created_at"].dt.floor("30min").dt.strftime("%H:%M")
                add_o = a.groupby(["date", "slot"], as_index=False).agg(add_orders=("order_id", "nunique"))
                grp = grp.merge(add_o, on=["date", "slot"], how="left")
            grp["add_orders"] = grp.get("add_orders", 0).fillna(0)
            grp["aov"] = grp["net"] / grp["orders"].replace(0, np.nan)
            grp["add_rate"] = grp["add_orders"] / grp["orders"].replace(0, np.nan)
            return grp

        def _render_heat(df: Optional[pd.DataFrame], key_prefix: str) -> None:
            if df is None or df.empty:
                st.info("无可用数据")
                return
            if metric == "订单数":
                mat = df.pivot(index="date", columns="slot", values="orders").fillna(0).astype(int)
            elif metric == "应收(优惠后)":
                mat = df.pivot(index="date", columns="slot", values="net").fillna(0.0)
            elif metric == "实收":
                mat = df.pivot(index="date", columns="slot", values="paid").fillna(0.0)
            elif metric == "客单(应收/订单)":
                mat = df.pivot(index="date", columns="slot", values="aov")
            else:
                mat = df.pivot(index="date", columns="slot", values="add_rate")
            cols = sorted(mat.columns, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))
            mat = mat[cols]

            view = st.radio("展示方式", options=["热力图", "渐变表格"], horizontal=True, key=f"heat_view_{key_prefix}_{metric}")
            if view == "渐变表格":
                st.dataframe(_simple_gradient_styler(mat), use_container_width=True)
            else:
                mdf = mat.reset_index().melt(id_vars="date", var_name="slot", value_name="value")
                chart = alt.Chart(mdf).mark_rect().encode(
                    x=alt.X("slot:N", title="半小时"),
                    y=alt.Y("date:N", title="日期"),
                    color=alt.Color("value:Q", title=metric),
                    tooltip=["date", "slot", "value"],
                ).properties(height=320)
                st.altair_chart(chart, use_container_width=True)

        if scope == "选中门店汇总":
            oall2 = pd.concat([filtered[s]["orders"] for s in sel_stores], ignore_index=True)
            pall2 = pd.concat([filtered[s]["pay"] for s in sel_stores], ignore_index=True)
            aall2 = pd.concat([filtered[s]["items_add"] for s in sel_stores], ignore_index=True)
            baseh = _build_heat(oall2, pall2, aall2)
            _render_heat(baseh, "all")

            st.markdown("### 动作建议（A2）：峰谷排班、促销时段、单加引导")
            if baseh is not None and not baseh.empty:
                agg = baseh.groupby("slot", as_index=False).agg(订单数=("orders", "sum"), 应收=("net", "sum"), 实收=("paid", "sum"), 单加订单=("add_orders", "sum"))
                agg["客单"] = agg["应收"] / agg["订单数"].replace(0, np.nan)
                agg["单加渗透率"] = agg["单加订单"] / agg["订单数"].replace(0, np.nan)
                peak = agg.sort_values("订单数", ascending=False).head(5)
                low = agg[agg["订单数"] > 0].sort_values("订单数", ascending=True).head(5)
                opp = agg[agg["订单数"] >= agg["订单数"].median()].sort_values("单加渗透率", ascending=True).head(5)

                st.dataframe(pd.concat([peak.assign(类型="峰值"), low.assign(类型="低谷"), opp.assign(类型="单加机会")], ignore_index=True), use_container_width=True)
                action = pd.concat([peak.assign(建议="峰值：加人/备货/保出餐"), low.assign(建议="低谷：促销/团购/引导单加"), opp.assign(建议="机会：强化单加话术")], ignore_index=True)
                key=_dl_key("ln963", sid) ,
                st.download_button("导出店长行动清单 CSV", data=action.to_csv(index=False).encode("utf-8-sig"), file_name="店长行动清单.csv", mime="text/csv")
            else:
                st.info("动作建议：当前范围无足够数据。")
        else:
            for sid in sel_stores:
                st.markdown(f"#### 门店 {sid}")
                dfh = _build_heat(filtered[sid]["orders"], filtered[sid]["pay"], filtered[sid]["items_add"])
                _render_heat(dfh, sid)

    # ⑪ 门店画像卡（两店对比）
    with tabs[10]:
        st.subheader("门店画像卡（两店对比）：一页判断“像谁 / 偏在哪 / 怎么改”")
        st.caption("选择两家门店进行对比：KPI对照 + 雷达图（能力结构）+ 偏离度拆解（规格/单加/支付/品类/时段）+ 行动建议。")

        if len(sel_stores) < 2:
            st.info("请在顶部“选择门店”中至少选择 2 家门店，才能进行两店对比。")
        else:
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                store_a = st.selectbox("门店A", options=sel_stores, index=0, key="portrait_a")
            with c2:
                opts_b = [s for s in sel_stores if s != store_a] or sel_stores
                store_b = st.selectbox("门店B", options=opts_b, index=0, key="portrait_b")
            with c3:
                bench_mode = st.selectbox("基准", options=["两店均值", "以门店A为标杆", "以门店B为标杆"], index=0, key="portrait_bench")

            fa = filtered[store_a]
            fb = filtered[store_b]

            sa = _compute_profile_scores(fa)
            sb = _compute_profile_scores(fb)

            kpi_cols = ["订单数", "应收", "实收", "客单", "退款率", "对账差异率", "单加渗透率", "单加金额占比", "规格多样性", "渠道多样性", "渠道最大占比", "品类Top5占比", "峰值Top3占比"]
            kpidf = pd.DataFrame({"指标": kpi_cols, store_a: [sa.get(k) for k in kpi_cols], store_b: [sb.get(k) for k in kpi_cols]})
            st.dataframe(kpidf, use_container_width=True)

            radar_dims = ["客单", "单加渗透率", "规格多样性", "渠道多样性", "品类Top5占比", "峰值Top3占比", "退款率", "对账差异率"]
            invert = {"品类Top5占比", "峰值Top3占比", "退款率", "对账差异率"}
            # 用“本次筛选门店的分布”做归一化，避免两店互相当标尺导致雷达图失真
            profiles = {}
            for _sid in sel_stores:
                try:
                    profiles[_sid] = _compute_profile_scores(filtered[_sid])
                except Exception:
                    profiles[_sid] = {}

            normalize_max = {}
            for d in radar_dims:
                vals = [float(profiles.get(_sid, {}).get(d, 0.0) or 0.0) for _sid in sel_stores]
                mx = float(np.nanpercentile(vals, 95)) if len(vals) else 1.0
                normalize_max[d] = max(mx, 1e-9)


            dfa = _radar_df(sa, radar_dims, normalize_max, invert); dfa["门店"] = store_a
            dfb = _radar_df(sb, radar_dims, normalize_max, invert); dfb["门店"] = store_b
            rdf = pd.concat([dfa, dfb], ignore_index=True)

            st.markdown("### 能力结构雷达图（0-1）：越外圈越强")
            st.altair_chart(_radar_chart(rdf, store_col="门店"), use_container_width=True)

            st.markdown("### 偏离度拆解（JS divergence）：告诉你“偏在哪”")

            def bench_dist(keys_a, vals_a, keys_b, vals_b):
                _, a, b = _align_two(keys_a, vals_a, keys_b, vals_b)
                if bench_mode == "以门店A为标杆":
                    q = a
                elif bench_mode == "以门店B为标杆":
                    q = b
                else:
                    q = 0.5 * (a + b)
                return a, b, q

            ma = _base_items(fa["items_main"]); mb = _base_items(fb["items_main"])
            spec_a = ma[ma["spec_norm"].notna()] if (ma is not None and not ma.empty and "spec_norm" in ma.columns) else pd.DataFrame()
            spec_b = mb[mb["spec_norm"].notna()] if (mb is not None and not mb.empty and "spec_norm" in mb.columns) else pd.DataFrame()
            k_sa, v_sa = _dist_from_group(spec_a, "spec_norm", "菜品数量", topn=10)
            k_sb, v_sb = _dist_from_group(spec_b, "spec_norm", "菜品数量", topn=10)
            va, vb, q = bench_dist(k_sa, v_sa, k_sb, v_sb)
            js_spec_a, js_spec_b = _js_divergence(va, q), _js_divergence(vb, q)

            k_aa, v_aa = _dist_from_group(fa["items_add"], "add_display", "amount", topn=15)
            k_ab, v_ab = _dist_from_group(fb["items_add"], "add_display", "amount", topn=15)
            va2, vb2, q2 = bench_dist(k_aa, v_aa, k_ab, v_ab)
            js_add_a, js_add_b = _js_divergence(va2, q2), _js_divergence(vb2, q2)

            k_pa, v_pa = _dist_from_group(fa["pay"], "支付类型", "总金额", topn=15)
            k_pb, v_pb = _dist_from_group(fb["pay"], "支付类型", "总金额", topn=15)
            va3, vb3, q3 = bench_dist(k_pa, v_pa, k_pb, v_pb)
            js_pay_a, js_pay_b = _js_divergence(va3, q3), _js_divergence(vb3, q3)

            def cat_dist(x):
                if x is None or x.empty or "categories" not in x.columns:
                    return [], np.array([], dtype=float)
                ex = x.copy().explode("categories"); ex["categories"] = ex["categories"].fillna("未分类")
                return _dist_from_group(ex, "categories", "优惠后小计价格", topn=25)

            k_ca, v_ca = cat_dist(fa["items_main"]); k_cb, v_cb = cat_dist(fb["items_main"])
            va4, vb4, q4 = bench_dist(k_ca, v_ca, k_cb, v_cb)
            js_cat_a, js_cat_b = _js_divergence(va4, q4), _js_divergence(vb4, q4)

            def slot_dist(o):
                if o is None or o.empty:
                    return [], np.array([], dtype=float)
                t = o.copy()
                t["slot"] = t["order_time"].dt.floor("30min").dt.strftime("%H:%M")
                return _dist_from_group(t, "slot", "dish_qty", topn=48)

            k_ta, v_ta = slot_dist(fa["orders"]); k_tb, v_tb = slot_dist(fb["orders"])
            va5, vb5, q5 = bench_dist(k_ta, v_ta, k_tb, v_tb)
            js_time_a, js_time_b = _js_divergence(va5, q5), _js_divergence(vb5, q5)

            dims = ["规格", "单加", "支付", "品类", "时段"]
            js_a = np.array([js_spec_a, js_add_a, js_pay_a, js_cat_a, js_time_a], dtype=float)
            js_b = np.array([js_spec_b, js_add_b, js_pay_b, js_cat_b, js_time_b], dtype=float)

            suma = float(js_a.sum()) if float(js_a.sum()) > 0 else 1.0
            sumb = float(js_b.sum()) if float(js_b.sum()) > 0 else 1.0

            div_df = pd.DataFrame({
                "维度": dims,
                f"{store_a} 偏离度": js_a,
                f"{store_a} 贡献": js_a / suma,
                f"{store_b} 偏离度": js_b,
                f"{store_b} 贡献": js_b / sumb,
            })
            st.dataframe(div_df.style.format({f"{store_a} 偏离度": "{:.4f}", f"{store_b} 偏离度": "{:.4f}", f"{store_a} 贡献": "{:.0%}", f"{store_b} 贡献": "{:.0%}"}), use_container_width=True)

            def top_reason(js_vec: np.ndarray) -> str:
                if float(js_vec.sum()) <= 0:
                    return "结构与基准几乎一致"
                k = int(np.argmax(js_vec))
                return f"主要偏离来自【{dims[k]}】（贡献 {js_vec[k]/js_vec.sum():.0%}）"

            colx, coly = st.columns(2)
            with colx:
                st.markdown(f"**{store_a} 结论：** {top_reason(js_a)}")
            with coly:
                st.markdown(f"**{store_b} 结论：** {top_reason(js_b)}")

            st.markdown("### 行动建议")
            def advice(scores: Dict[str, float], radar_scores: pd.DataFrame) -> List[str]:
                # 先跑“硬规则”告警，再补“短板Top2”建议，保证每家店都有可执行动作
                out: List[str] = []

                if scores.get("退款率", 0) > 0.03:
                    out.append("退款偏高：优先排查 Top退款菜品与对应时段，核查出品稳定性/配送问题/支付对账口径。")
                if scores.get("对账差异率", 0) > 0.02:
                    out.append("对账差异偏大：重点看混合支付与团购渠道，排查漏记/退款口径差/跨日支付。")
                if scores.get("渠道最大占比", 0) > 0.75:
                    out.append("渠道过度依赖：注意单一渠道波动风险，优化多渠道渗透（团购/小程序/外卖）。")

                weakest = radar_scores.sort_values("得分", ascending=True).head(2)["维度"].tolist()
                for w in weakest:
                    if w == "单加渗透率":
                        out.append("单加偏弱：高峰时段强化“加料话术/提示卡”，重点提升单加-鸡丁/单加-卤蛋等渗透。")
                    elif w == "规格多样性":
                        out.append("规格结构偏单一：检查主食结构（细面/宽面/米饭/宽粉/无需主食/套餐标准）是否失衡，尝试用套餐与陈列引导分流。")
                    elif w == "渠道多样性":
                        out.append("支付渠道过于单一：检查线上/线下、团购渗透与支付方式覆盖，减少单点波动。")
                    elif w == "品类Top5占比":
                        out.append("品类过度集中：检查爆品依赖与缺货风险，用套餐/第二爆品/搭配单加分流。")
                    elif w == "峰值Top3占比":
                        out.append("峰值过度集中：在Top3半小时加人/备货；低谷用团购、小套餐、主食组合拉平波动。")
                    elif w == "客单":
                        out.append("客单偏低：用套餐标准、单加推荐与高毛利饮品/小食提升客单。")
                    elif w == "退款率":
                        out.append("退款偏高：优先看退款热力时段与Top退款菜品，定位流程或渠道问题。")
                    elif w == "对账差异率":
                        out.append("对账差异偏大：重点核对支付表与订单表时间口径、跨日支付与退款归属。")

                seen = set()
                uniq = []
                for s in out:
                    if s not in seen:
                        uniq.append(s); seen.add(s)
                return uniq[:6] if uniq else ["结构健康：建议作为标杆门店，沉淀可复制打法（菜单结构/单加/渠道/排班）。"]

            a1, a2 = st.columns(2)
            with a1:
                st.markdown(f"**{store_a} 建议**")
                for s in advice(sa, dfa):
                    st.write("• " + s)
            with a2:
                st.markdown(f"**{store_b} 建议**")
                for s in advice(sb, dfb):
                    st.write("• " + s)

            st.download_button(
                "导出画像卡指标对比 CSV",
                data=kpidf.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"门店画像卡对比_{store_a}_vs_{store_b}.csv",
                mime="text/csv",
                key=_dl_key("portrait"),
            )




if __name__ == "__main__":
    main()