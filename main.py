# -*- coding: utf-8 -*-
"""
main.py
一鍵流程：
1) 蒐集帳戶名單：acct_alert 全部 + ~N 個非 alert 帳戶
2) 以 get_history.extract_histories() 產出 ./training_data/random_data.csv
3) 呼叫 training.train_entry() 訓練 MLP，輸出最佳模型與紀錄

需求檔案：
- ./dataset/acct_transaction.csv（或容忍誤拼 ./dataset/acct_transcation.csv）
- ./dataset/acct_alert.csv
- 已有的 get_history.py、training_dataset.py、training.py、model.py
"""

import os
import argparse
import random
import pandas as pd

# 從既有腳本直接引入可用函式（不額外模塊化）
from get_history import detect_txn_path, extract_histories
from training_dataset import collect_non_alert_accounts
from training import train_entry

DEFAULT_TXN   = "./dataset/acct_transaction.csv"
DEFAULT_ALERT = "./dataset/acct_alert.csv"
OUT_RANDOM    = "./training_data/random_data.csv"
OUT_MODEL_DIR = "./output/model_run"

def build_random_training_csv(txn_path: str,
                              alert_path: str,
                              out_csv: str,
                              extra_accts: int = 10000,
                              seed: int = 42,
                              from_col: str = "from_acct",
                              to_col: str = "to_acct") -> str:
    """
    產出 training_data/random_data.csv
    - 目標帳戶 = alert 全部 + 約 N 個非 alert 帳戶（單次掃描交易檔蒐集）
    - 用 extract_histories 擷取所有相關交易（整行）
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    txn_path = detect_txn_path(txn_path)

    if not os.path.exists(alert_path):
        raise FileNotFoundError(f"alert file not found: {alert_path}")
    alert_df = pd.read_csv(alert_path, usecols=["acct"])
    alert_set = set(alert_df["acct"].astype(str))

    extra_list = collect_non_alert_accounts(
        txn_path=txn_path,
        alert_set=alert_set,
        target_n=extra_accts,
        from_col=from_col,
        to_col=to_col
    )

    targets = list(alert_set) + extra_list
    print(f"[main] targets -> alert: {len(alert_set)}, extra: {len(extra_list)}, total: {len(targets)}")

    df = extract_histories(
        txn_path=txn_path,
        accts=targets,
        from_col=from_col,
        to_col=to_col,
        chunksize=200_000
    )
    print(f"[main] matched rows in transactions: {len(df)}")

    df.to_csv(out_csv, index=False)
    print(f"[main] saved training csv: {out_csv}")
    return out_csv

def parse_list_of_ints(text: str):
    return [int(x) for x in text.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="One-click: sample -> build random_data.csv -> train model -> save best ckpt")
    # Step1: 取樣/建檔
    ap.add_argument("--txn", default=None, help=f"Path to acct_transaction.csv (default tries {DEFAULT_TXN})")
    ap.add_argument("--alert", default=DEFAULT_ALERT)
    ap.add_argument("--out_random", default=OUT_RANDOM)
    ap.add_argument("--extra_accts", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from_col", default="from_acct")
    ap.add_argument("--to_col", default="to_acct")

    # Step2: 訓練
    ap.add_argument("--out_dir", default=OUT_MODEL_DIR)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--cat_bucket_size", type=int, default=50000)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")  # 會在 training.py 中 fallback 到 cpu if not available
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cat_emb_dim", type=int, default=16)
    ap.add_argument("--bn", action="store_true")

    args = ap.parse_args()

    # 1) 建 random_data.csv
    random_csv = build_random_training_csv(
        txn_path=args.txn,
        alert_path=args.alert,
        out_csv=args.out_random,
        extra_accts=args.extra_accts,
        seed=args.seed,
        from_col=args.from_col,
        to_col=args.to_col
    )

    # 2) 訓練
    hidden = parse_list_of_ints(args.hidden)
    best = train_entry(
        input_csv=random_csv,
        alert_csv=args.alert,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cat_bucket_size=args.cat_bucket_size,
        hidden=hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device if os.environ.get("CUDA_VISIBLE_DEVICES", "") != " -1" else "cpu",
        dropout=args.dropout,
        cat_emb_dim=args.cat_emb_dim,
        bn=args.bn
    )
    print(f"[main] training finished. best model: {best}")

if __name__ == "__main__":
    main()
