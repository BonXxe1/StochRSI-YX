#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""沪深主板 StochRSI 选股器（含 Moomoo 个股详情链接）。

筛选口径：
- 沪深主板；剔除 ST/*ST；
- Wilder RSI(15)；RSI 的 Stochastic 窗口 8；
- 原始 StochRSI 经 SMA(10) 得到 K，再经 SMA(3) 得到 D；
- 只保留 K < 10 且最新报告期归母净利润、扣非净利润均为正的股票；
- 按扣非净利润降序排列。

说明：
- 行情和财报来自原脚本使用的新浪、腾讯、东方财富公开接口；
- Moomoo 链接使用公开 HTTPS 个股详情页路由，不使用未公开的私有 URL Scheme；
- 手机是否直接唤起 Moomoo App 由系统的链接关联设置决定；否则会打开网页个股详情页。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np

# ========================= 参数 =========================
RSI_PERIOD = 15
STOCH_PERIOD = 8
K_SMOOTH = 10
D_SMOOTH = 3
K_THRESHOLD = 10.0
HISTORY_KEEP = 220
DEFAULT_WORKERS = 10
QFQ_WORKERS_RETRY = 6
PROFIT_WORKERS = 6
QUOTE_BATCH_SIZE = 60
HTTP_TIMEOUT = 18
MAX_RETRIES = 3
MIN_SH_STOCKS = 1500
MIN_SZ_STOCKS = 1300
MIN_TODAY_COVERAGE = 0.50

OUT_DIR = Path(os.environ.get("STOCK_OUT_DIR", "out"))
OUT_CSV = OUT_DIR / "latest.csv"
OUT_JSON = OUT_DIR / "summary.json"
OUT_STATUS = OUT_DIR / "status.txt"

SINA_NODE_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
SINA_QFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/qianfuquan.js"
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
EASTMONEY_FINANCE_URL = "https://datacenter.eastmoney.com/securities/api/data/get"
MOOMOO_STOCK_URL = "https://www.moomoo.com/hans/stock/{code}-{market}"

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}
PAT_QFQ = re.compile(r"_(\d{4})_(\d{2})_(\d{2}):\s*\"([+-]?[0-9]+(?:\.[0-9]+)?)\"")
SH_TZ = ZoneInfo("Asia/Shanghai")

Entry = Tuple[str, float]


# ========================= 通用工具 =========================
def log(message: str) -> None:
    print(message, flush=True)


def http_get(url: str, timeout: int = HTTP_TIMEOUT, headers: Optional[Dict[str, str]] = None) -> bytes:
    req_headers = dict(UA)
    if headers:
        req_headers.update(headers)
    req = Request(url, headers=req_headers)
    with urlopen(req, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"HTTP status {status}: {url}")
        return response.read()


def retry_delay(attempt: int) -> None:
    time.sleep(min(0.5 * (2 ** attempt), 4.0))


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_status(state: str) -> None:
    atomic_write_text(OUT_STATUS, state.rstrip() + "\n")


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "" or value == "-":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def moomoo_market(code: str) -> str:
    if code.startswith("6"):
        return "SH"
    if code.startswith(("000", "001", "002", "003")):
        return "SZ"
    raise ValueError(f"非沪深主板股票代码：{code}")


def moomoo_symbol(code: str) -> str:
    return f"{moomoo_market(code)}.{code}"


def moomoo_url(code: str) -> str:
    market = moomoo_market(code)
    return MOOMOO_STOCK_URL.format(code=code, market=market)


# ========================= 技术指标 =========================
def simple_sma(values: Sequence[float], period: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if period <= 0:
        raise ValueError("SMA 周期必须大于 0")
    if len(x) < period:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, period)
    valid = np.isfinite(windows).all(axis=1)
    target = out[period - 1 :]
    target[valid] = windows[valid].mean(axis=1)
    return out


def wilder_rsi(closes: Sequence[float], period: int = RSI_PERIOD) -> np.ndarray:
    prices = np.asarray(closes, dtype=float)
    rsi = np.full(prices.shape, np.nan, dtype=float)
    if period <= 0:
        raise ValueError("RSI 周期必须大于 0")
    if len(prices) <= period:
        return rsi
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("收盘价必须全部为有限正数")

    delta = np.diff(prices)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())

    def rsi_value(gain: float, loss: float) -> float:
        # 价格完全横盘时采用中性 RSI=50，避免 0/0 被误判为 RSI=100。
        if gain == 0 and loss == 0:
            return 50.0
        if loss == 0:
            return 100.0
        if gain == 0:
            return 0.0
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    rsi[period] = rsi_value(avg_gain, avg_loss)
    for index in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[index]) / period
        avg_loss = (avg_loss * (period - 1) + losses[index]) / period
        rsi[index + 1] = rsi_value(avg_gain, avg_loss)
    return rsi


def stochrsi(
    closes: Sequence[float],
    rsi_period: int = RSI_PERIOD,
    stoch_period: int = STOCH_PERIOD,
    k_period: int = K_SMOOTH,
    d_period: int = D_SMOOTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 K、D、RSI，K/D 均为 0–100 标度。"""
    rsi = wilder_rsi(closes, rsi_period)
    raw = np.full(rsi.shape, np.nan, dtype=float)
    if len(rsi) >= stoch_period:
        windows = np.lib.stride_tricks.sliding_window_view(rsi, stoch_period)
        valid = np.isfinite(windows).all(axis=1)
        highs = np.full(len(windows), np.nan, dtype=float)
        lows = np.full(len(windows), np.nan, dtype=float)
        highs[valid] = windows[valid].max(axis=1)
        lows[valid] = windows[valid].min(axis=1)
        high_full = np.full(rsi.shape, np.nan, dtype=float)
        low_full = np.full(rsi.shape, np.nan, dtype=float)
        high_full[stoch_period - 1 :] = highs
        low_full[stoch_period - 1 :] = lows
        denom = high_full - low_full
        valid_now = np.isfinite(denom) & np.isfinite(rsi)
        flat = valid_now & (np.abs(denom) < 1e-10)
        normal = valid_now & ~flat
        # 零振幅窗口沿用原脚本的中性值 50，不人为制造超卖信号。
        raw[flat] = 50.0
        raw[normal] = ((rsi[normal] - low_full[normal]) / denom[normal]) * 100.0
    k = simple_sma(raw, k_period)
    d = simple_sma(k, d_period)
    return k, d, rsi


# ========================= 股票池 =========================
def is_main_board(code: str) -> bool:
    return code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def is_st(name: str) -> bool:
    compact = re.sub(r"\s+", "", name or "").upper()
    return "ST" in compact


def normalize_symbol(raw_symbol: Any, code: str) -> str:
    expected = ("sh" if code.startswith("6") else "sz") + code
    raw = str(raw_symbol or "").strip().lower()
    match = re.search(r"(?:sh|sz)\d{6}", raw)
    if match and match.group(0).endswith(code):
        return match.group(0)
    return expected


def fetch_node(node: str) -> List[dict]:
    stocks: List[dict] = []
    max_pages = 80
    for page in range(1, max_pages + 1):
        params = urlencode({"page": page, "num": 100, "sort": "symbol", "asc": 1, "node": node})
        url = f"{SINA_NODE_URL}?{params}"
        data = None
        last_error: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = http_get(url, timeout=20)
                data = json.loads(raw.decode("utf-8-sig"))
                if data is None:
                    data = []
                if not isinstance(data, list):
                    raise ValueError(f"{node} 第 {page} 页返回格式不是列表")
                break
            except Exception as exc:  # 上游接口异常后重试
                last_error = exc
                if attempt < MAX_RETRIES:
                    retry_delay(attempt)
        if data is None:
            raise RuntimeError(f"{node} 第 {page} 页失败：{last_error}")
        if not data:
            if page == 1:
                raise RuntimeError(f"{node} 第 1 页为空，股票列表接口可能异常")
            break
        stocks.extend(row for row in data if isinstance(row, dict))
        if len(data) < 100:
            break
    else:
        raise RuntimeError(f"{node} 分页超过 {max_pages} 页，停止以防止不完整股票池")
    return stocks


# ========================= 历史行情、实时行情 =========================
def fetch_qfq(symbol: str) -> Tuple[str, Optional[List[Entry]], Optional[str]]:
    url = SINA_QFQ_URL.format(symbol=symbol)
    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = http_get(url, timeout=HTTP_TIMEOUT)
            text = raw.decode("utf-8", "replace").lstrip("\ufeff")
            pairs = PAT_QFQ.findall(text)
            by_date: Dict[str, float] = {}
            for year, month, day, raw_close in pairs:
                date_text = f"{year}-{month}-{day}"
                try:
                    # 严格检查日期和价格，跳过异常记录，不让单条脏数据丢弃整只股票。
                    datetime.strptime(date_text, "%Y-%m-%d")
                    close = float(raw_close)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(close) and close > 0:
                    by_date[date_text] = close
            if len(by_date) < 40:
                return symbol, None, f"历史K线不足40根（{len(by_date)}）"
            rows = sorted(by_date.items())[-HISTORY_KEEP:]
            return symbol, rows, None
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                retry_delay(attempt)
    return symbol, None, last_error or "历史行情抓取失败"


def parse_tencent_quotes(text: str) -> Dict[str, dict]:
    quotes: Dict[str, dict] = {}
    for line in text.split(";"):
        if '="' not in line:
            continue
        left, right = line.split('="', 1)
        symbol = left.split("_")[-1].strip().lower()
        body = right.rsplit('"', 1)[0]
        parts = body.split("~")
        # 腾讯行情协议字段：3 最新价、4 昨收、6 成交量、30 行情时间。
        if len(parts) <= 30:
            continue
        px = to_float(parts[3])
        prev = to_float(parts[4])
        volume = to_float(parts[6])
        timestamp = parts[30].strip()
        if (
            not re.fullmatch(r"(?:sh|sz)\d{6}", symbol)
            or px is None
            or prev is None
            or volume is None
            or not re.fullmatch(r"\d{14}", timestamp)
        ):
            continue
        quotes[symbol] = {"px": px, "prev": prev, "vol": volume, "ts": timestamp}
    return quotes


def fetch_quotes(symbols: Sequence[str]) -> Tuple[Dict[str, dict], List[str]]:
    out: Dict[str, dict] = {}
    failed_batches: List[str] = []
    for start in range(0, len(symbols), QUOTE_BATCH_SIZE):
        chunk = symbols[start : start + QUOTE_BATCH_SIZE]
        url = TENCENT_QUOTE_URL + ",".join(chunk)
        success = False
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                raw = http_get(url, timeout=15)
                text = raw.decode("gbk", "replace")
                parsed = parse_tencent_quotes(text)
                if not parsed and text.strip():
                    raise ValueError("行情响应无法解析")
                out.update(parsed)
                success = True
                break
            except Exception as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES - 1:
                    retry_delay(attempt)
        if not success:
            failed_batches.append(f"{start}-{start + len(chunk) - 1}: {last_error}")
    return out, failed_batches


# ========================= 财务过滤 =========================
def fetch_latest_profit(code: str) -> Optional[dict]:
    """返回最新报告期归母净利润、扣非净利润（单位：元）。"""
    suffix = "SH" if code.startswith("6") else "SZ"
    filt = f'(SECUCODE="{code}.{suffix}")'
    params = {
        "type": "RPT_F10_FINANCE_MAINFINADATA",
        "sty": "REPORT_DATE_NAME,KCFJCXSYJLR,PARENTNETPROFIT",
        "filter": filt,
        "p": 1,
        "ps": 1,
        "sr": -1,
        "st": "REPORT_DATE",
        "source": "HSF10",
        "client": "PC",
    }
    url = f"{EASTMONEY_FINANCE_URL}?{urlencode(params)}"
    headers = {"Referer": "https://data.eastmoney.com/", "Accept": "application/json, text/plain, */*"}
    last_error: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = http_get(url, timeout=15, headers=headers)
            data = json.loads(raw.decode("utf-8-sig"))
            rows = ((data.get("result") or {}).get("data") or [])
            if not rows:
                return None
            row = rows[0]
            return {
                "kcf": to_float(row.get("KCFJCXSYJLR")),
                "parent": to_float(row.get("PARENTNETPROFIT")),
                "report": str(row.get("REPORT_DATE_NAME") or row.get("REPORT_DATE") or ""),
            }
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                retry_delay(attempt)
    raise RuntimeError(f"{code} 财报接口失败：{last_error}")


def fetch_profits(codes: Sequence[str]) -> Tuple[Dict[str, Optional[dict]], Dict[str, str]]:
    results: Dict[str, Optional[dict]] = {}
    errors: Dict[str, str] = {}
    if not codes:
        return results, errors
    with ThreadPoolExecutor(max_workers=PROFIT_WORKERS) as executor:
        futures = {executor.submit(fetch_latest_profit, code): code for code in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                results[code] = future.result()
            except Exception as exc:
                errors[code] = str(exc)
    return results, errors


# ========================= 飞书推送（可点击 Moomoo 链接） =========================
def push_feishu(hits: Sequence[dict], summary: dict) -> None:
    webhook = os.environ.get("FEISHU_WEBHOOK", "").strip()
    if not webhook:
        log("未配置 FEISHU_WEBHOOK，跳过飞书推送")
        return

    date_text = summary.get("date", "")
    title = f"StochRSI 主板筛选 | {date_text} | K<{K_THRESHOLD:g} | {len(hits)} 只"
    content: List[List[dict]] = []
    if not hits:
        content.append([{"tag": "text", "text": "本次没有符合条件的股票。"}])
    else:
        content.append([{
            "tag": "text",
            "text": (
                f"主板去 ST；归母及扣非净利润均为正；按扣非净利润降序。"
                f"\n有效当日行情覆盖率：{summary.get('today_coverage_pct', 0):.1f}%"
            ),
        }])
        content.append([{"tag": "text", "text": "代码 / 名称 | K / D | 扣非净利润 | 详情："}])
        for hit in hits[:40]:
            pct = hit.get("pct")
            pct_text = f" {pct:+.2f}%" if pct is not None else ""
            profit_yi = hit["kcf"] / 1e8
            line = (
                f"{hit['code']} {hit['name']} | K={hit['K']:.2f} D="
                f"{hit['D'] if hit['D'] is not None else '-'} | "
                f"扣非{profit_yi:.2f}亿{pct_text} "
            )
            content.append([
                {"tag": "text", "text": line},
                {"tag": "a", "text": "打开 Moomoo", "href": hit["moomoo_url"]},
            ])
        if len(hits) > 40:
            content.append([{"tag": "text", "text": f"其余 {len(hits) - 40} 只请查看 out/latest.csv。"}])

    content.append([{
        "tag": "text",
        "text": (
            f"候选 K<10：{summary.get('k_below_10', 0)}；"
            f"真实亏损剔除：{summary.get('dropped_loss', 0)}；"
            f"财报缺失/异常剔除：{summary.get('dropped_unknown', 0)}；"
            f"耗时：{summary.get('elapsed_sec', 0)}s。"
        ),
    }])
    payload = {
        "msg_type": "post",
        "content": {"post": {"zh_cn": {"title": title, "content": content}}},
    }
    request = Request(
        webhook,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8", "replace")
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"HTTP {response.status}: {body[:300]}")
            try:
                result = json.loads(body) if body else {}
            except json.JSONDecodeError:
                result = {}
            if isinstance(result, dict) and result.get("code", 0) not in (0, None):
                raise RuntimeError(f"飞书返回错误：{result}")
            log(f"飞书推送成功 HTTP {response.status}")
    except Exception as exc:
        log(f"飞书推送失败：{exc}")


# ========================= 输出 =========================
def write_outputs(hits: Sequence[dict], summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 同时提供公式超链接和原始 URL，兼容 Excel/WPS 及其他数据处理工具。
    csv_fd, csv_tmp = tempfile.mkstemp(prefix="latest.", suffix=".tmp", dir=str(OUT_DIR))
    try:
        with os.fdopen(csv_fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "代码", "名称", "Moomoo证券代码", "Moomoo详情", "Moomoo详情链接",
                "扣非净利润元", "归母净利润元", "报告期", "K", "D", "RSI",
                "最新价", "涨跌幅%", "计算日期",
            ])
            for hit in hits:
                url = hit["moomoo_url"]
                formula = f'=HYPERLINK("{url}","打开 Moomoo")'
                writer.writerow([
                    hit["code"], hit["name"], hit["moomoo_symbol"], formula, url,
                    f"{hit['kcf']:.2f}", f"{hit['parent_np']:.2f}", hit["report"],
                    f"{hit['K']:.2f}", "" if hit["D"] is None else f"{hit['D']:.2f}",
                    "" if hit["rsi"] is None else f"{hit['rsi']:.2f}",
                    "" if hit["px"] is None else f"{hit['px']:.3f}",
                    "" if hit["pct"] is None else f"{hit['pct']:.2f}", hit["date"],
                ])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(csv_tmp, OUT_CSV)
    except Exception:
        try:
            os.unlink(csv_tmp)
        except OSError:
            pass
        raise

    json_content = json.dumps(
        {"summary": summary, "hits": list(hits)}, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    atomic_write_text(OUT_JSON, json_content)


# ========================= 选股流程 =========================
def build_universe(sh_rows: Sequence[dict], sz_rows: Sequence[dict]) -> Tuple[Dict[str, dict], int]:
    universe: Dict[str, dict] = {}
    st_skipped = 0
    for row in list(sh_rows) + list(sz_rows):
        raw_code = str(row.get("code") or "").strip()
        if not raw_code.isdigit():
            continue
        code = raw_code.zfill(6)
        if not is_main_board(code):
            continue
        name = re.sub(r"\s+", "", str(row.get("name") or ""))
        if is_st(name):
            st_skipped += 1
            continue
        symbol = normalize_symbol(row.get("symbol"), code)
        universe[symbol] = {"code": code, "name": name}
    return universe, st_skipped


def latest_hist_date(series: Dict[str, List[Entry]]) -> str:
    dates = [rows[-1][0] for rows in series.values() if rows]
    if not dates:
        raise RuntimeError("没有可用历史行情")
    # 取出现次数最多的末根K线日期，避免少量异常股票日期影响基准。
    counts = Counter(dates)
    # 若票数相同，固定选择较新的日期，避免线程完成顺序影响结果。
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def calculate_candidates(
    universe: Dict[str, dict],
    series: Dict[str, List[Entry]],
    quotes: Dict[str, dict],
    today: str,
    market_hist_date: str,
) -> Tuple[List[dict], List[float], Dict[str, int]]:
    hits: List[dict] = []
    all_k: List[float] = []
    date_counts: Dict[str, int] = {}

    for symbol, rows in series.items():
        if not rows:
            continue
        dates = [item[0] for item in rows]
        closes = [item[1] for item in rows]
        last_date = dates[-1]
        quote = quotes.get(symbol)
        quote_is_today = bool(
            quote
            and quote["ts"].startswith(today.replace("-", ""))
            and quote["px"] > 0
            and quote["prev"] > 0
            and quote["vol"] > 0
        )
        current_px: Optional[float] = None
        current_prev: Optional[float] = None

        if last_date == today:
            # 新浪日K已包含当日数据，直接使用，避免重复追加一根K线。
            effective_date = today
            if quote_is_today and quote["px"] > 0 and quote["prev"] > 0:
                current_px, current_prev = quote["px"], quote["prev"]
            else:
                current_px = None
                current_prev = None
        elif (
            last_date == market_hist_date
            and quote_is_today
            and quote["px"] > 0
            and quote["prev"] > 0
            and quote["vol"] > 0
        ):
            # 新浪历史K线尚未落当日数据时，用腾讯当日价相对昨收的比例，
            # 乘以前复权末收盘价，构造同一复权口径的当日收盘/盘中价。
            closes.append(closes[-1] * (quote["px"] / quote["prev"]))
            effective_date = today
            current_px, current_prev = quote["px"], quote["prev"]
        else:
            # 不再拿不同日期的历史K线混入本次当日优选。
            continue

        if not closes or not np.isfinite(closes[-1]) or closes[-1] <= 0:
            continue
        date_counts[effective_date] = date_counts.get(effective_date, 0) + 1
        k_values, d_values, rsi_values = stochrsi(closes)
        if not len(k_values) or not np.isfinite(k_values[-1]):
            continue
        k_value = float(k_values[-1])
        d_value = float(d_values[-1]) if len(d_values) and np.isfinite(d_values[-1]) else None
        rsi_value = float(rsi_values[-1]) if np.isfinite(rsi_values[-1]) else None
        all_k.append(k_value)

        if k_value >= K_THRESHOLD:
            continue

        pct = None
        if current_px is not None and current_prev is not None and current_prev > 0:
            pct = round((current_px / current_prev - 1.0) * 100.0, 2)

        code = universe[symbol]["code"]
        hits.append({
            "code": code,
            "name": universe[symbol]["name"],
            "moomoo_symbol": moomoo_symbol(code),
            "moomoo_url": moomoo_url(code),
            "K": round(k_value, 2),
            "D": None if d_value is None else round(d_value, 2),
            "rsi": None if rsi_value is None else round(rsi_value, 2),
            "px": None if current_px is None else round(current_px, 3),
            "pct": pct,
            "date": effective_date,
        })
    return hits, all_k, date_counts


def run_pipeline(workers: int = DEFAULT_WORKERS, no_feishu: bool = False) -> dict:
    today = datetime.now(SH_TZ).strftime("%Y-%m-%d")
    today_compact = today.replace("-", "")
    started = time.time()
    log(f"交易日期基准：{today}（Asia/Shanghai）")

    log("获取沪市、深市股票列表...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_sh = executor.submit(fetch_node, "sh_a")
        future_sz = executor.submit(fetch_node, "sz_a")
        sh_rows = future_sh.result()
        sz_rows = future_sz.result()
    log(f"原始列表：沪市 {len(sh_rows)}，深市 {len(sz_rows)}")

    universe, st_skipped = build_universe(sh_rows, sz_rows)
    symbols = sorted(universe)
    sh_count = sum(1 for info in universe.values() if info["code"].startswith("6"))
    sz_count = len(universe) - sh_count
    log(f"主板去ST后：{len(symbols)}；沪市 {sh_count}；深市 {sz_count}；剔除ST {st_skipped}")
    if sh_count < MIN_SH_STOCKS or sz_count < MIN_SZ_STOCKS:
        raise RuntimeError(
            f"股票列表完整性检查未通过：沪市 {sh_count}/{MIN_SH_STOCKS}，"
            f"深市 {sz_count}/{MIN_SZ_STOCKS}。为防止错误结果，不生成新文件。"
        )

    log("下载前复权历史行情...")
    series: Dict[str, List[Entry]] = {}
    errors: Dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_qfq, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, rows, error = future.result()
            except Exception as exc:
                sym, rows, error = symbol, None, str(exc)
            done += 1
            if error or not rows:
                errors[sym] = error or "历史行情为空"
            else:
                series[sym] = rows
            if done % 400 == 0 or done == len(symbols):
                log(f"历史行情 {done}/{len(symbols)}：成功 {len(series)}，失败 {len(errors)}")

    if errors:
        retry_symbols = list(errors)
        log(f"重试历史行情失败项：{len(retry_symbols)} 只")
        time.sleep(1.0)
        with ThreadPoolExecutor(max_workers=QFQ_WORKERS_RETRY) as executor:
            for sym, rows, error in executor.map(fetch_qfq, retry_symbols):
                if error or not rows:
                    errors[sym] = error or "历史行情为空"
                else:
                    series[sym] = rows
                    errors.pop(sym, None)
        log(f"历史行情重试后：成功 {len(series)}，失败 {len(errors)}")

    if len(series) < len(symbols) * 0.90:
        raise RuntimeError(
            f"前复权历史行情覆盖不足：{len(series)}/{len(symbols)}，"
            "低于90%，为防止样本偏差不生成新结果。"
        )

    log("获取腾讯行情...")
    quotes, quote_batch_errors = fetch_quotes(list(series))
    fresh_quotes = sum(1 for quote in quotes.values() if quote["ts"].startswith(today_compact))
    market_hist_date = latest_hist_date(series)
    today_qfq = sum(1 for rows in series.values() if rows[-1][0] == today)
    coverage = sum(
        1 for symbol, rows in series.items()
        if rows[-1][0] == today
        or (
            rows[-1][0] == market_hist_date
            and symbol in quotes
            and quotes[symbol]["ts"].startswith(today_compact)
            and quotes[symbol]["px"] > 0
            and quotes[symbol]["prev"] > 0
            and quotes[symbol]["vol"] > 0
        )
    )
    coverage_pct = coverage / max(len(series), 1) * 100.0
    log(
        f"腾讯行情 {len(quotes)} 只，当日时间戳 {fresh_quotes} 只；"
        f"新浪当日K线 {today_qfq} 只；可用于当日计算 {coverage}/{len(series)} ({coverage_pct:.1f}%)"
    )
    if coverage_pct < MIN_TODAY_COVERAGE * 100.0:
        state = (
            f"skip: 当日行情覆盖率 {coverage_pct:.1f}% 低于 "
            f"{MIN_TODAY_COVERAGE * 100:.0f}%；历史结果未覆盖"
        )
        log(state)
        write_status(state)
        return {"status": "skip", "reason": state}

    hits, all_k, bar_dates = calculate_candidates(
        universe, series, quotes, today, market_hist_date
    )
    if not all_k:
        raise RuntimeError("没有计算出有效 StochRSI，停止写入结果")
    if bar_dates.get(today, 0) < len(all_k) * 0.95:
        raise RuntimeError(
            f"当日K线日期不一致：{bar_dates}；有效计算样本 {len(all_k)}，停止写入结果。"
        )

    raw_hit_count = len(hits)
    log(f"当日有效计算 {len(all_k)} 只；K<{K_THRESHOLD:g} 候选 {raw_hit_count} 只")

    profit_codes = [hit["code"] for hit in hits]
    profits, profit_errors = fetch_profits(profit_codes)
    kept: List[dict] = []
    dropped_loss = 0
    dropped_unknown = 0
    for hit in hits:
        info = profits.get(hit["code"])
        if not info or info.get("kcf") is None or info.get("parent") is None:
            dropped_unknown += 1
            continue
        kcf = info["kcf"]
        parent = info["parent"]
        if kcf <= 0 or parent <= 0:
            dropped_loss += 1
            continue
        hit["kcf"] = float(kcf)
        hit["parent_np"] = float(parent)
        hit["report"] = info.get("report", "")
        kept.append(hit)

    kept.sort(key=lambda item: (-item["kcf"], item["code"]))
    log(
        f"财报筛选后 {len(kept)} 只；真实亏损剔除 {dropped_loss}；"
        f"财报缺失/异常剔除 {dropped_unknown}；接口异常 {len(profit_errors)}"
    )

    k_array = np.asarray(all_k, dtype=float)
    summary = {
        "date": today,
        "status": "ok",
        "algo": (
            f"Wilder RSI({RSI_PERIOD}) + Stoch({STOCH_PERIOD}) + "
            f"SMA(K,{K_SMOOTH}) + SMA(D,{D_SMOOTH})"
        ),
        "threshold": f"K < {K_THRESHOLD:g}",
        "universe": len(symbols),
        "computed": int(k_array.size),
        "history_ok": len(series),
        "history_errors": len(errors),
        "history_error_sample": list(errors.items())[:8],
        "quote_count": len(quotes),
        "fresh_quote_count": fresh_quotes,
        "quote_batch_errors": quote_batch_errors,
        "today_qfq_count": today_qfq,
        "today_coverage": coverage,
        "today_coverage_pct": round(coverage_pct, 2),
        "market_hist_date": market_hist_date,
        "st_skipped": st_skipped,
        "k_below_10": raw_hit_count,
        "dropped_loss": dropped_loss,
        "dropped_unknown": dropped_unknown,
        "profit_api_errors": len(profit_errors),
        "profit_error_sample": list(profit_errors.items())[:8],
        "below_10": len(kept),
        "below_20": int((k_array < 20).sum()),
        "above_80": int((k_array > 80).sum()),
        "k_min": round(float(k_array.min()), 2),
        "k_p50": round(float(np.percentile(k_array, 50)), 2),
        "k_max": round(float(k_array.max()), 2),
        "bar_dates": bar_dates,
        "elapsed_sec": round(time.time() - started, 1),
    }

    write_outputs(kept, summary)
    if not no_feishu:
        push_feishu(kept, summary)
    write_status("ok")
    log(f"结果已写入：{OUT_CSV}；JSON：{OUT_JSON}；优选 {len(kept)} 只")
    return {"status": "ok", "summary": summary, "hits": kept}


# ========================= 离线自检 =========================
def self_test() -> None:
    flat = np.full(80, 10.0)
    flat_rsi = wilder_rsi(flat)
    assert np.isclose(flat_rsi[RSI_PERIOD], 50.0)
    flat_k, flat_d, _ = stochrsi(flat)
    assert np.isfinite(flat_k[-1]) and np.isclose(flat_k[-1], 50.0)
    assert np.isfinite(flat_d[-1]) and np.isclose(flat_d[-1], 50.0)

    rising = np.arange(1.0, 100.0)
    rising_rsi = wilder_rsi(rising)
    assert np.isclose(rising_rsi[-1], 100.0)
    falling = np.arange(100.0, 1.0, -1.0)
    falling_rsi = wilder_rsi(falling)
    assert np.isclose(falling_rsi[-1], 0.0)

    assert is_main_board("600519") and is_main_board("000858")
    assert not is_main_board("300750") and not is_main_board("688001")
    assert is_st("贵州茅台ST") and is_st("*ST某公司") and not is_st("贵州茅台")
    assert moomoo_symbol("600519") == "SH.600519"
    assert moomoo_symbol("000858") == "SZ.000858"
    assert moomoo_url("600519") == "https://www.moomoo.com/hans/stock/600519-SH"
    assert moomoo_url("000858") == "https://www.moomoo.com/hans/stock/000858-SZ"

    fields = [""] * 31
    fields[1] = "贵州茅台"
    fields[3] = "1500.00"
    fields[4] = "1490.00"
    fields[6] = "12000"
    fields[30] = "20261109150000"
    parsed = parse_tencent_quotes(f'v_sh600519="{"~".join(fields)}";')
    assert parsed["sh600519"]["px"] == 1500.0
    assert parsed["sh600519"]["ts"] == "20261109150000"
    assert to_float("NaN") is None and to_float("12.5") == 12.5

    # 确认 CSV 写入时，名称含逗号也能完整保留为一个字段。
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["000001", "名称,含逗号", "https://example.com"])
    parsed_row = next(csv.reader(io.StringIO(buffer.getvalue())))
    assert parsed_row[1] == "名称,含逗号"
    log("SELF-TEST PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="沪深主板 StochRSI 选股器 + Moomoo 个股链接")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="历史行情并发数，默认 10")
    parser.add_argument("--no-feishu", action="store_true", help="本次不推送飞书")
    parser.add_argument("--self-test", action="store_true", help="仅运行离线自检，不访问网络")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise SystemExit("--workers 必须在 1 到 32 之间")
    if args.self_test:
        self_test()
        return 0
    try:
        run_pipeline(workers=args.workers, no_feishu=args.no_feishu)
        return 0
    except Exception as exc:
        message = f"failed: {type(exc).__name__}: {exc}"
        log(message)
        try:
            write_status(message)
        except Exception as status_exc:
            log(f"写入失败状态也失败：{status_exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
