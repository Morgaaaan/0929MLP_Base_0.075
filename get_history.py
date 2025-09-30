#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
get_history.py
- 依帳戶（acct）擷取其所有相關交易（from_acct 或 to_acct），輸出整行 row。
- 支援命令列使用；也提供一個單純函式 `extract_histories(...)` 讓其它腳本直接 import 呼叫。
- 容忍交易檔誤拼名：acct_transaction.csv / acct_transcation.csv 都會嘗試。
"""

import os
import argparse
from typing import Iterable, List, Set, Optional
import pandas as pd

DEFAULT_TXN = "./dataset/acct_transaction.csv"
ALT_TXN     = "./dataset/acct_transcation.csv"  # 容忍誤拼

def detect_txn_path(user_path: Optional[str]) -> str:
    if user_path and os.path.exists(user_path):
        return user_path
    if os.path.exists(DEFAULT_TXN):
        return DEFAULT_TXN
    if os.path.exists(ALT_TXN):
        return ALT_TXN
    raise FileNotFoundError(
        f"Transaction file not found. Tried: {user_path or '(none)'}, {DEFAULT_TXN}, {ALT_TXN}"
    )

def read_acct_list(accts_arg: Optional[List[str]], acct_file: Optional[str]) -> List[str]:
    accts: List[str] = []
    if accts_arg:
        for a in accts_arg:
            accts.extend([x.strip() for x in a.split(",") if x.strip()])
    if acct_file:
        if not os.path.exists(acct_file):
            raise FileNotFoundError(f"acct list file not found: {acct_file}")
        df = pd.read_csv(acct_file)
        col = "acct" if "acct" in df.columns else None
        if col is None:
            # 若沒有明確 'acct' 欄，嘗試找出一個字串欄位
            cand = [c for c in df.columns if df[c].dtype == object]
            if len(cand) == 1:
                col = cand[0]
            else:
                for c in cand:
                    if "acct" in c.lower():
                        col = c
                        break
                if col is None and cand:
                    col = cand[0]
        if col is None:
            raise ValueError("Could not infer account column from file; please ensure it has an 'acct' column.")
        accts.extend(df[col].astype(str).tolist())
    # 去重去空
    accts = [a for a in {x.strip() for x in accts} if a]
    if not accts:
        raise ValueError("No accounts provided. Use --acct and/or --acct-file.")
    return accts

def extract_histories(
    txn_path: str,
    accts: Iterable[str],
    from_col: str = "from_acct",
    to_col: str = "to_acct",
    chunksize: int = 200_000
) -> pd.DataFrame:
    """
    依據帳戶清單，擷取其所有相關交易（整行）。
    """
    txn_path = detect_txn_path(txn_path)
    targets: Set[str] = {str(a) for a in accts}
    if not targets:
        raise ValueError("Empty account set.")

    # 檢查欄位
    sample = pd.read_csv(txn_path, nrows=1, low_memory=False)
    if from_col not in sample.columns or to_col not in sample.columns:
        raise ValueError(f"Required columns missing in {txn_path}: need '{from_col}' and '{to_col}'")

    all_cols = list(sample.columns)
    dfs = []
    for chunk in pd.read_csv(
        txn_path, chunksize=chunksize, low_memory=False,
        dtype={from_col: str, to_col: str}
    ):
        mask = chunk[from_col].isin(targets) | chunk[to_col].isin(targets)
        if mask.any():
            dfs.append(chunk.loc[mask, all_cols])

    if not dfs:
        return pd.DataFrame(columns=all_cols)
    return pd.concat(dfs, ignore_index=True)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Extract full-row transaction histories for given accounts.")
    ap.add_argument("--txn", default=None, help=f"Path to acct_transaction.csv (default tries {DEFAULT_TXN} then {ALT_TXN})")
    ap.add_argument("--from-col", default="from_acct")
    ap.add_argument("--to-col", default="to_acct")
    ap.add_argument("--chunksize", type=int, default=200_000)

    ap.add_argument("--acct", action="append", help="Account id(s). Can be given multiple times or comma-separated.")
    ap.add_argument("--acct-file", default=None, help="CSV with an 'acct' column (or a single string column).")
    ap.add_argument("--out", default=None, help="Path to write a single merged CSV of all matched rows.")

    args = ap.parse_args()

    txn_path = detect_txn_path(args.txn)
    accts = read_acct_list(args.acct, args.acct_file)
    print(f"[get_history] txn: {txn_path}")
    print(f"[get_history] target accounts: {len(accts)}")

    df = extract_histories(
        txn_path=txn_path,
        accts=accts,
        from_col=args.from_col,
        to_col=args.to_col,
        chunksize=args.chunksize
    )
    print(f"[get_history] matched rows: {len(df)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[get_history] saved: {args.out}")
    else:
        print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
