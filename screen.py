#!/usr/bin/env python3
"""沪深主板 StochRSI 筛选，对齐 moomoo 的 STOCHRSI。

RSI 周期 15（Wilder）
STOCH 周期 8
K = 原始值的 10 日简单移动平均
D = K 的 3 日简单移动平均
去掉 ST，再去掉最新报告期亏损股（归母净利润或扣非净利润 ≤ 0）。
K < 10 的结果按扣非净利润从高到低排序，推送到飞书。
"""
import json
import os
import re
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
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
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn",
}
PAT = re.compile(r"_(\d{4})_(\d{2})_(\d{2}):\"([0-9.]+)\"")
SH_TZ = ZoneInfo("Asia/Shanghai")


def write_status(state):
    os.makedirs("out", exist_ok=True)
    with open(OUT_STATUS, "w", encoding="utf-8") as f:
        f.write(state + "\n")


def http_get(url, timeout=18):
    req = Request(url, headers=UA)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def simple_sma(x, period):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if len(x) < period:
        return out
    window = np.lib.stride_tricks.sliding_window_view(x, period)
    mask = ~np.isnan(window).any(axis=1)
    out[period - 1 :][mask] = window[mask].mean(axis=1)
    return out


def wilder_rsi(closes, n):
    closes = np.asarray(closes, dtype=float)
    rsi = np.full(closes.shape, np.nan)
    if len(closes) <= n:
        return rsi
    delta = np.diff(closes)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = float(gain[:n].mean())
    avg_loss = float(loss[:n].mean())

    def value(ag, al):
        if al == 0:
            return 100.0
        if ag == 0:
            return 0.0
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)

    rsi[n] = value(avg_gain, avg_loss)
    for i in range(n, len(delta)):
        avg_gain = (avg_gain * (n - 1) + gain[i]) / n
        avg_loss = (avg_loss * (n - 1) + loss[i]) / n
        rsi[i + 1] = value(avg_gain, avg_loss)
    return rsi


def stochrsi(closes):
    rsi = wilder_rsi(closes, N)
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
    flat = valid & (np.abs(denom) < 1e-10)
    ok = valid & ~flat
    raw[flat] = 50.0
    raw[ok] = (rsi[ok] - low[ok]) / denom[ok] * 100.0
    k = simple_sma(raw, P1)
    d = simple_sma(k, P2)
    return k, d, rsi


def is_main_board(code):
    return code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def is_st(name):
    return "ST" in (name or "").upper().replace(" ", "")


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


def fetch_latest_profit(code):
    """最新报告期归母净利润、扣非净利润，单位元。"""
    suffix = "SH" if code.startswith("6") else "SZ"
    filt = f'(SECUCODE="{code}.{suffix}")'
    url = (
        "https://datacenter.eastmoney.com/securities/api/data/get"
        "?type=RPT_F10_FINANCE_MAINFINADATA"
        "&sty=REPORT_DATE_NAME,KCFJCXSYJLR,PARENTNETPROFIT"
        f"&filter={quote(filt)}&p=1&ps=1&sr=-1&st=REPORT_DATE"
    )
    req = Request(
        url,
        headers={
            "User-Agent": UA["User-Agent"],
            "Referer": "https://data.eastmoney.com/",
        },
    )
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = (data.get("result") or {}).get("data") or []
    if not rows:
        return None
    row = rows[0]
    return {
        "kcf": row.get("KCFJCXSYJLR"),
        "parent": row.get("PARENTNETPROFIT"),
        "report": row.get("REPORT_DATE_NAME") or "",
    }


def push_feishu(hits, summary):
    webhook = os.environ.get("FEISHU_WEBHOOK", "").strip()
    if not webhook:
        print("未配置 FEISHU_WEBHOOK，跳过飞书推送", flush=True)
        return
    date = summary.get("date", "")
    if not hits:
        text = (
            f"【StochRSI】{date} 收盘筛选完成\n"
            f"主板去ST、去亏损后 K < 10：0 只\n"
            f"K 中位数 {summary.get('k_p50')}"
        )
    else:
        lines = [
            f"【StochRSI】{date} · 主板去ST · K<10且扣非盈利 · "
            f"按扣非净利润从高到低 共 {len(hits)} 只",
            "",
        ]
        show = hits[:40]
        for h in show:
            pct = h.get("pct")
            pct_s = f"{pct:+.2f}%" if pct is not None else "-"
            yi = h["kcf"] / 1e8
            lines.append(
                f"{h['code']} {h['name']}  扣非{yi:.2f}亿({h['report']})  "
                f"K={h['K']}  D={h['D']}  {h.get('px')}  {pct_s}"
            )
        if len(hits) > len(show):
            lines.append(f"... 其余 {len(hits) - len(show)} 只见 out/latest.csv")
        lines += [
            "",
            f"K<10 共 {summary.get('k_below_10')} 只，亏损或无财报剔除 "
            f"{summary.get('dropped_loss')} 只 · 用时 {summary.get('elapsed_sec')}s",
        ]
        text = "\n".join(lines)
    body = json.dumps(
        {"msg_type": "text", "content": {"text": text}},
        ensure_ascii=False,
    ).encode("utf-8")
    req = Request(
        webhook,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            print(f"飞书推送状态 {resp.status}", flush=True)
    except Exception as e:
        print(f"飞书推送失败: {e}", flush=True)


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
    st_skipped = 0
    for row in sh + sz:
        code = str(row.get("code") or "").zfill(6)
        if not is_main_board(code):
            continue
        name = (row.get("name") or "").replace(" ", "")
        if is_st(name):
            st_skipped += 1
            continue
        symbol = row.get("symbol") or (("sh" if code.startswith("6") else "sz") + code)
        universe[symbol] = {"code": code, "name": name}
    symbols = sorted(universe)
    sh_n = sum(1 for s in symbols if universe[s]["code"].startswith("6"))
    sz_n = len(symbols) - sh_n
    print(f"main(no ST) {len(symbols)} sh {sh_n} sz {sz_n} skipped_ST {st_skipped}", flush=True)
    if sh_n < 1500 or sz_n < 1300:
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
    if today_bars < max(len(ks), 1) * 0.5:
        print(f"多数股票没有今天的K线 {bar_dates}。不覆盖上次结果。", flush=True)
        write_status("skip")
        return

    k_below_10 = len(hits)
    print(f"fetch profit for {k_below_10} hits...", flush=True)
    profits = {}
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(fetch_latest_profit, h["code"]): h["code"] for h in hits}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                profits[code] = fut.result()
            except Exception as e:
                profits[code] = None
                print(f"profit {code} failed: {e}", flush=True)

    kept = []
    dropped_loss = 0
    for h in hits:
        info = profits.get(h["code"])
        kcf = None if not info else info.get("kcf")
        parent = None if not info else info.get("parent")
        if kcf is None or parent is None or kcf <= 0 or parent <= 0:
            dropped_loss += 1
            continue
        h["kcf"] = float(kcf)
        h["parent_np"] = float(parent)
        h["report"] = info["report"]
        kept.append(h)
    hits = kept
    hits.sort(key=lambda row: (-row["kcf"], row["code"]))
    print(f"after profit filter {len(hits)} dropped {dropped_loss}", flush=True)

    arr = np.asarray(ks, dtype=float)
    summary = {
        "date": today,
        "algo": "moomoo STOCHRSI Wilder RSI + SMA(K,10) + SMA(D,3)",
        "universe": len(symbols),
        "computed": int(arr.size),
        "errors": len(errors),
        "error_sample": list(errors.items())[:8],
        "st_skipped": st_skipped,
        "k_below_10": k_below_10,
        "dropped_loss": dropped_loss,
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
        f.write("代码,名称,扣非净利润元,归母净利润元,报告期,K,D,RSI,最新价,涨跌幅%,日期\n")
        for h in hits:
            f.write(
                f"{h['code']},{h['name']},{h['kcf']:.2f},{h['parent_np']:.2f},{h['report']},"
                f"{h['K']},{h['D']},{h['rsi']},{h['px']},{h['pct']},{h['date']}\n"
            )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "hits": hits}, f, ensure_ascii=False, indent=2)
    push_feishu(hits, summary)
    write_status("ok")
    print(f"wrote {OUT_CSV} hits={len(hits)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED {exc}", file=sys.stderr, flush=True)
        raise
