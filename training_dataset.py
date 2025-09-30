#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_dataset.py (revised)
- Alert 帳戶：從 ./dataset/acct_transaction.csv 擷取其所有相關交易
- 非 Alert 帳戶：從 ./sort_data/txn_neither_in_alert.csv 擷取（只在此檔內）其所有相關交易
- 非 Alert 帳戶名單：於 txn_neither_in_alert.csv 中抽取約 N 個帳戶（避開 alert_set）
- 輸出：./output/training_data/random_data.csv
- 結束時在終端顯示總行數
"""

import os
import argparse
from typing import Set, List
import pandas as pd

# 直接使用 get_history.py 的函式（可指向任意 CSV，有 from_acct/to_acct 即可）
from get_history import detect_txn_path, extract_histories

# 預設路徑
TXN_FULL_DEFAULT = "./dataset/acct_transaction.csv"
TXN_NONALERT_DEFAULT = "./sort_data/txn_neither_in_alert.csv"
ALERT_DEFAULT = "./dataset/acct_alert.csv"
OUT_DEFAULT = "./output/training_data/random_data.csv"

def collect_accounts_from_file(txn_path: str,
                               exclude_set: Set[str],
                               target_n: int,
                               from_col: str = "from_acct",
                               to_col: str = "to_acct") -> List[str]:
    """
    從指定交易檔（例如 txn_neither_in_alert.csv）中蒐集帳戶，排除 exclude_set（alert 帳戶），
    直到約 target_n 或檔案掃完。
    """
    accs: Set[str] = set()
    # 檢查欄位
    head = pd.read_csv(txn_path, nrows=1, low_memory=False)
    for c in (from_col, to_col):
        if c not in head.columns:
            raise ValueError(f"Missing column '{c}' in {txn_path}")
    # 分塊讀取以控記憶體
    for chunk in pd.read_csv(txn_path, chunksize=200_000, low_memory=False, dtype={from_col: str, to_col: str}):
        accs.update(set(chunk[from_col].astype(str)))
        accs.update(set(chunk[to_col].astype(str)))
        # 排除 alert
        accs.difference_update(exclude_set)
        if len(accs) >= target_n:
            break
    # 截取到目標數量（保序不可保證，此處無關緊要）
    acc_list = list(accs)
    if len(acc_list) > target_n:
        acc_list = acc_list[:target_n]
    return acc_list

def main():
    ap = argparse.ArgumentParser(description="Build random_data.csv using alert from full txn, non-alert from txn_neither_in_alert.csv")
    ap.add_argument("--txn_full", default=TXN_FULL_DEFAULT, help="Full transaction CSV for alert accounts")
    ap.add_argument("--txn_nonalert", default=TXN_NONALERT_DEFAULT, help="Non-alert transaction CSV for non-alert accounts")
    ap.add_argument("--alert", default=ALERT_DEFAULT)
    ap.add_argument("--extra_accts", type=int, default=10000, help="~Number of non-alert accounts to sample from txn_neither_in_alert.csv")
    ap.add_argument("--from_col", default="from_acct")
    ap.add_argument("--to_col", default="to_acct")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    # 準備輸出資料夾
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 讀取 alert 帳戶
    if not os.path.exists(args.alert):
        raise FileNotFoundError(f"alert file not found: {args.alert}")
    alert_df = pd.read_csv(args.alert, usecols=["acct"])
    alert_set: Set[str] = set(alert_df["acct"].astype(str))
    print(f"[build] alert accounts: {len(alert_set)}")

    # 檔案存在檢查 / 自動偵測
    txn_full_path = detect_txn_path(args.txn_full)
    if not os.path.exists(args.txn_nonalert):
        raise FileNotFoundError(f"non-alert transaction file not found: {args.txn_nonalert}")
    txn_nonalert_path = args.txn_nonalert  # 不需要 detect，直接用指定檔

    # 從非 alert 交易檔抽樣帳戶（排除 alert）
    nonalert_accts: List[str] = collect_accounts_from_file(
        txn_path=txn_nonalert_path,
        exclude_set=alert_set,
        target_n=args.extra_accts,
        from_col=args.from_col,
        to_col=args.to_col
    )
    print(f"[build] sampled non-alert accounts from txn_neither_in_alert.csv: {len(nonalert_accts)}")

    # 擷取 alert 帳戶歷史（從完整交易檔）
    print("[build] extracting alert histories from full transaction file ...")
    alert_hist = extract_histories(
        txn_path=txn_full_path,
        accts=list(alert_set),
        from_col=args.from_col,
        to_col=args.to_col,
        chunksize=200_000
    )

    # 擷取非 alert 帳戶歷史（僅從 non-alert 檔中）
    print("[build] extracting non-alert histories from txn_neither_in_alert.csv ...")
    nonalert_hist = extract_histories(
        txn_path=txn_nonalert_path,
        accts=nonalert_accts,
        from_col=args.from_col,
        to_col=args.to_col,
        chunksize=200_000
    )

    # 合併並輸出
    print("[build] concatenating ...")
    if alert_hist is None or alert_hist.empty:
        combined = nonalert_hist
    elif nonalert_hist is None or nonalert_hist.empty:
        combined = alert_hist
    else:
        combined = pd.concat([alert_hist, nonalert_hist], ignore_index=True)

    combined.to_csv(args.out, index=False)
    print(f"[build] saved: {args.out}")
    print(f"[build] total rows: {len(combined)}")  # <=== 終端列印總行數

if __name__ == "__main__":
    main()
