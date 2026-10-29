#!/usr/bin/env python3
"""沪深主板 StochRSI 筛选。通达信公式，前复权。

N=15  M=8  P1=10  P2=3
K = SMA(STOCHRSI_RAW, P1, 1)
D = SMA(K, P2, 1)
通达信 SMA(X,N,1) = (X + (N-1)*昨日) / N
输出 K < 10 的股票。非交易日直接退出，不覆盖上次结果。
"""
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

N, M, P1, P2 = 15, 8, 10, 3
THRESHOLD = 10.0
KEEP = 220
WORKERS = 12
OUT_CSV = "out/latest.csv"
OUT_JSON = "out/summary.json"
OUT_STATUS = "out/status.txt"
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
PAT = re.compile(r"_(\d{4})_(\d{2})_(\d{2}):\"([0-9.]+)\"")
SH_TZ = ZoneInfo("Asia/Shanghai")


def write_status(state):
    os.makedirs("out", exist_ok=True)
    with open(OUT_STATUS, "w", encoding="utf-8") as f:
        f.write(state + "\n")


def http_get(url, timeout=18):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def tdx_sma(x, n, m):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    prev = None
    for i, xi in enumerate(x):
        if np.isnan(xi):
            continue
        prev = float(xi) if prev is None else (m * float(xi) + (n - m) * prev) / n
        out[i] = prev
    return out


def stochrsi(closes):
    closes = np.asarray(closes, dtype=float)
    diff = np.empty_like(closes)
    diff[0] = np.nan
    diff[1:] = closes[1:] - closes[:-1]
    up = np.where(np.isnan(diff), np.nan, np.maximum(diff, 0.0))
    ab = np.where(np.isnan(diff), np.nan, np.abs(diff))
    sma_up = tdx_sma(up, N, 1)
    sma_ab = tdx_sma(ab, N, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = sma_up / sma_ab * 100.0
    rsi[(sma_ab == 0) | np.isnan(sma_ab)] = np.nan
    high = np.full(len(closes), np.nan)
    low = np.full(len(closes), np.nan)
    if len(closes) >= M:
        window = np.lib.stride_tricks.sliding_window_view(rsi, M)
        mask = ~np.isnan(window).any(axis=1)
        high[M - 1 :][mask] = window[mask].max(axis=1)
        low[M - 1 :][mask] = window[mask].min(axis=1)
    raw = np.full(len(closes), np.nan)
    denom = high - low
    valid = ~np.isnan(denom) & ~np.isnan(rsi)
    zero = valid & (np.abs(denom) < 1e-10)
    ok = valid & ~zero
    raw[zero] = 50.0
    raw[ok] = (rsi[ok] - low[ok]) / denom[ok] * 100.0
    k = tdx_sma(raw, P1, 1)
    d = tdx_sma(k, P2, 1)
    return k, d, rsi


def is_main_board(code):
    return code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def fetch_node(node):
    stocks = []
    page = 1
    empty_tries = 0
    while page <= 80:
        url = (
            "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            f"Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1&node={node}"
        )
        data = None
        err = None
        for attempt in range(4):
            try:
                data = json.loads(http_get(url, timeout=20).decode("utf-8"))
                break
            except Exception as e:
                err = e
                time.sleep(0.4 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"{node} page {page}: {err}")
        if not data:
            empty_tries += 1
            if empty_tries >= 5:
                raise RuntimeError(f"empty page {node} {page}")
            time.sleep(0.8)
            continue
        empty_tries = 0
        stocks.extend(data)
        if len(data) < 100:
            break
        page += 1
    return stocks


def fetch_qfq(symbol):
    url = f"https://finance.sina.com.cn/realstock/company/{symbol}/qianfuquan.js"
    last = None
    for attempt in range(3):
        try:
            text = http_get(url, timeout=18).decode("utf-8", "replace")
            pairs = PAT.findall(text)
            if len(pairs) < 40:
                return symbol, None, "short"
            rows = [(f"{y}-{m}-{d}", float(v)) for y, m, d, v in pairs[: KEEP + 5]]
            rows.sort()
            return symbol, rows[-KEEP:], None
        except Exception as e:
            last = str(e)
            time.sleep(0.4 * (attempt + 1))
    return symbol, None, last


def fetch_quotes(symbols):
    out = {}
    for i in range(0, len(symbols), 60):
        chunk = symbols[i : i + 60]
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        try:
            text = http_get(url, timeout=15).decode("gbk", "replace")
        except Exception as e:
            print(f"quote batch {i} failed: {e}", flush=True)
            continue
        for line in text.split(";"):
            if '="' not in line:
                continue
            sym = line.split('="', 1)[0].split("_")[-1].strip()
            body = line.split('="', 1)[1].rsplit('"', 1)[0]
            parts = body.split("~")
            if len(parts) < 7:
                continue
            try:
                px = float(parts[3])
                prev = float(parts[4])
                vol = float(parts[6] or 0)
            except ValueError:
                continue
            ts = next((x for x in parts if len(x) == 14 and x.isdigit()), "")
            out[sym] = {"px": px, "prev": prev, "vol": vol, "ts": ts}
    return out


def main():
    today = datetime.now(SH_TZ).strftime("%Y-%m-%d")
    today_compact = today.replace("-", "")
    t0 = time.time()
    print(f"today {today} Asia/Shanghai", flush=True)

    print("lists...", flush=True)
    with ThreadPoolExecutor(2) as ex:
        sh = ex.submit(fetch_node, "sh_a").result()
        sz = ex.submit(fetch_node, "sz_a").result()
    print(f"sh_a {len(sh)} sz_a {len(sz)}", flush=True)

    universe = {}
    for row in sh + sz:
        code = str(row.get("code") or "").zfill(6)
        if not is_main_board(code):
            continue
        symbol = row.get("symbol") or (("sh" if code.startswith("6") else "sz") + code)
        universe[symbol] = {
            "code": code,
            "name": (row.get("name") or "").replace(" ", ""),
        }
    symbols = sorted(universe)
    sh_n = sum(code.startswith("6") for code in (universe[s]["code"] for s in symbols))
    sz_n = len(symbols) - sh_n
    print(f"main {len(symbols)} sh {sh_n} sz {sz_n}", flush=True)
    if sh_n < 1600 or sz_n < 1400:
        raise SystemExit(f"股票列表不完整 sh={sh_n} sz={sz_n}")

    series = {}
    errors = {}
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_qfq, s) for s in symbols]
        for fut in as_completed(futs):
            sym, rows, err = fut.result()
            done += 1
            if err or not rows:
                errors[sym] = err or "empty"
            else:
                series[sym] = rows
            if done % 400 == 0 or done == len(symbols):
                print(
                    f"qfq {done}/{len(symbols)} ok={len(series)} err={len(errors)} t={time.time()-t0:.0f}s",
                    flush=True,
                )
    if errors:
        retry = list(errors)
        print(f"retry {len(retry)}", flush=True)
        errors = {}
        time.sleep(1.5)
        with ThreadPoolExecutor(6) as ex:
            for sym, rows, err in ex.map(fetch_qfq, retry):
                if err or not rows:
                    errors[sym] = err or "empty"
                else:
                    series[sym] = rows
        print(f"after retry ok={len(series)} err={len(errors)}", flush=True)

    if len(series) < len(symbols) * 0.9:
        raise SystemExit(f"前复权下载失败过多 ok={len(series)} err={len(errors)}")

    print("quotes...", flush=True)
    quotes = fetch_quotes(list(series))
    fresh = sum(1 for q in quotes.values() if q["ts"].startswith(today_compact))
    print(f"quotes {len(quotes)} dated_today {fresh}", flush=True)
    if len(quotes) >= 500 and fresh < len(quotes) * 0.5:
        print("今天不是交易日，或收盘行情还没更新。不覆盖上次结果。", flush=True)
        write_status("skip")
        return

    hits = []
    ks = []
    bar_dates = {}
    for sym, rows in series.items():
        dates = [r[0] for r in rows]
        closes = [r[1] for r in rows]
        q = quotes.get(sym)
        if (
            q
            and q["px"] > 0
            and q["prev"] > 0
            and q["vol"] > 0
            and q["ts"].startswith(today_compact)
            and dates[-1] < today
        ):
            closes.append(closes[-1] * (q["px"] / q["prev"]))
            dates.append(today)
        k, d, rsi = stochrsi(closes)
        if np.isnan(k[-1]):
            continue
        kv = float(k[-1])
        dv = float(d[-1]) if not np.isnan(d[-1]) else None
        ks.append(kv)
        bar_dates[dates[-1]] = bar_dates.get(dates[-1], 0) + 1
        if kv >= THRESHOLD:
            continue
        px = q["px"] if q and q["px"] > 0 else None
        prev = q["prev"] if q and q["prev"] > 0 else None
        pct = round((px / prev - 1) * 100, 2) if px and prev else None
        hits.append(
            {
                "code": universe[sym]["code"],
                "name": universe[sym]["name"],
                "K": round(kv, 2),
                "D": None if dv is None else round(dv, 2),
                "rsi": None if np.isnan(rsi[-1]) else round(float(rsi[-1]), 2),
                "px": px,
                "pct": pct,
                "date": dates[-1],
            }
        )

    today_bars = bar_dates.get(today, 0)
    if today_bars < len(ks) * 0.5:
        print(f"多数股票没有今天的K线 {bar_dates}。不覆盖上次结果。", flush=True)
        write_status("skip")
        return

    hits.sort(key=lambda row: (row["K"], row["code"]))
    arr = np.asarray(ks, dtype=float)
    summary = {
        "date": today,
        "universe": len(symbols),
        "computed": int(arr.size),
        "errors": len(errors),
        "error_sample": list(errors.items())[:8],
        "below_10": len(hits),
        "below_20": int((arr < 20).sum()),
        "above_80": int((arr > 80).sum()),
        "k_min": round(float(arr.min()), 2),
        "k_p50": round(float(np.percentile(arr, 50)), 2),
        "k_max": round(float(arr.max()), 2),
        "bar_dates": bar_dates,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)

    os.makedirs("out", exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig") as f:
        f.write("代码,名称,K,D,RSI,最新价,涨跌幅%,日期\n")
        for h in hits:
            f.write(
                f"{h['code']},{h['name']},{h['K']},{h['D']},{h['rsi']},{h['px']},{h['pct']},{h['date']}\n"
            )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "hits": hits}, f, ensure_ascii=False, indent=2)
    write_status("ok")
    print(f"wrote {OUT_CSV} hits={len(hits)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED {exc}", file=sys.stderr, flush=True)
        raise
