# -*- coding: utf-8 -*-
"""
inference.py (two-stage output)
1) 從 ./dataset/acct_predict.csv 讀 acct 名單（保序）
2) 用 get_history.extract_histories 抓該帳戶所有相關交易（from 或 to 命中）
3) 載入 ./output/model_run/best_model.pt 與 feature_spec.json
4) 對每筆交易推論，取「正類機率 prob_1」
5) 以帳戶聚合 -> 取該帳戶所有相關交易 prob_1 的平均 -> 輸出 ./output/predict_raw.csv（acct, score）
6) 以門檻（預設 0.5）把 score 轉成 0/1 -> 輸出 ./output/prediction.csv（acct, label）

!!! 調整門檻請改這一行（非常重要） !!!
THRESH = 0.5
"""

import os
import argparse
import json
from typing import Dict, List
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import MLPClassifier
from get_history import extract_histories, detect_txn_path

# ====== <<< 調整二分類門檻就在這行 >>> ======
THRESH = 0.25
# ==========================================

def stable_bucket(s: str, bucket: int, salt: str = "") -> int:
    key = (salt + "||" + s).encode("utf-8", errors="ignore")
    return int(hashlib.md5(key).hexdigest(), 16) % bucket

class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, spec: Dict):
        self.df = df.reset_index(drop=True)
        self.numeric_cols: List[str] = spec.get("numeric_cols", [])
        self.cat_cols: List[str] = spec.get("cat_cols", [])
        self.cat_buckets: Dict[str, int] = spec.get("cat_buckets", {})

        # 補欄：numeric -> 0.0；cat -> "nan"
        for c in self.numeric_cols:
            if c not in self.df.columns:
                self.df[c] = 0.0
        for c in self.cat_cols:
            if c not in self.df.columns:
                self.df[c] = "nan"

        self.Xnum = self.df[self.numeric_cols].astype(float).values if self.numeric_cols else np.zeros((len(self.df), 0), dtype=float)
        self.Xcat = {}
        for c in self.cat_cols:
            v = self.df[c].astype(str).fillna("nan").values
            bucket = self.cat_buckets.get(c, 50000)
            self.Xcat[c] = np.array([stable_bucket(x, bucket, salt=c) for x in v], dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        x_num = torch.tensor(self.Xnum[idx], dtype=torch.float32)
        x_cat = {c: torch.tensor(self.Xcat[c][idx], dtype=torch.long) for c in self.cat_cols}
        return x_num, x_cat

def load_model_and_spec(ckpt_path: str, spec_path: str, device: str):
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    state = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(spec.get("num_classes", 2))
    if num_classes != 2:
        raise ValueError(f"Current inference expects binary model (num_classes==2), but got {num_classes}.")
    model = MLPClassifier(
        input_num_dim=len(spec.get("numeric_cols", [])),
        cat_buckets=spec.get("cat_buckets", {}),
        num_classes=num_classes,
        hidden=state.get("hparams", {}).get("hidden", [256, 128]),
        dropout=state.get("hparams", {}).get("dropout", 0.2),
        cat_emb_dim=state.get("hparams", {}).get("cat_emb_dim", 16),
        bn=state.get("hparams", {}).get("bn", True),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, spec

def main():
    ap = argparse.ArgumentParser(description="Inference: acct_predict.csv -> histories -> model -> raw & thresholded outputs")
    ap.add_argument("--predict_csv", default="./dataset/acct_predict.csv")
    ap.add_argument("--txn", default=None, help="Path to acct_transaction.csv (auto-detect if not set)")
    ap.add_argument("--model_ckpt", default="./output/model_run/best_model.pt")
    ap.add_argument("--spec_path", default="./output/model_run/feature_spec.json")
    ap.add_argument("--raw_csv", default="./output/predict_raw.csv")
    ap.add_argument("--out_csv", default="./output/prediction.csv")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from_col", default="from_acct")
    ap.add_argument("--to_col", default="to_acct")
    args = ap.parse_args()

    # 1) 讀取 acct 名單（保序）
    if not os.path.exists(args.predict_csv):
        raise FileNotFoundError(f"predict csv not found: {args.predict_csv}")
    pred_df = pd.read_csv(args.predict_csv)
    if "acct" not in pred_df.columns:
        raise ValueError("acct_predict.csv must contain column 'acct'")
    acct_list = pred_df["acct"].astype(str).tolist()

    # 2) 擷取相關交易
    txn_path = detect_txn_path(args.txn)
    hist_df = extract_histories(
        txn_path=txn_path,
        accts=acct_list,
        from_col=args.from_col,
        to_col=args.to_col,
        chunksize=200_000
    )

    # 若完全沒有交易被匹配，直接輸出 raw=0.0、label=0
    if hist_df.empty:
        os.makedirs(os.path.dirname(args.raw_csv), exist_ok=True)
        pd.DataFrame({"acct": acct_list, "score": [0.0] * len(acct_list)}).to_csv(args.raw_csv, index=False)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        pd.DataFrame({"acct": acct_list, "label": [0] * len(acct_list)}).to_csv(args.out_csv, index=False)
        print(f"[inference] no history rows; wrote raw zeros to {args.raw_csv} and labels to {args.out_csv}")
        return

    # 3) 載入模型與 spec（強制二分類）
    model, spec = load_model_and_spec(args.model_ckpt, args.spec_path, args.device)
    hist_df[args.from_col] = hist_df[args.from_col].astype(str)
    hist_df[args.to_col]   = hist_df[args.to_col].astype(str)

    # 4) 準備推論資料
    ds = InferenceDataset(hist_df, spec)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 5) 批次推論 -> 逐列 prob_1
    row_probs = []
    with torch.no_grad():
        for x_num, x_cat in dl:
            x_num = x_num.to(args.device)
            x_cat = {k: v.to(args.device) for k, v in x_cat.items()}
            logits = model(x_num, x_cat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            row_probs.append(probs)
    probs_all = np.vstack(row_probs)  # [N,2]
    prob1 = probs_all[:, 1]           # 正類機率

    # 6) 以帳戶聚合 -> 取平均 prob_1 作為該帳戶 score
    from_vals = hist_df[args.from_col].to_numpy()
    to_vals   = hist_df[args.to_col].to_numpy()
    acct2score: Dict[str, float] = {}
    for a in acct_list:
        mask = (from_vals == a) | (to_vals == a)
        if mask.any():
            acct2score[a] = float(prob1[mask].mean())
        else:
            acct2score[a] = 0.0  # 沒有任何交易時給 0.0

    # 7) 輸出 RAW 浮點檔（acct,score）
    raw_df = pd.DataFrame({"acct": acct_list, "score": [acct2score[a] for a in acct_list]})
    os.makedirs(os.path.dirname(args.raw_csv), exist_ok=True)
    raw_df.to_csv(args.raw_csv, index=False)
    print(f"[inference] raw scores saved: {args.raw_csv}")

    # 8) 以門檻轉 0/1（!!! 門檻請改檔頭 THRESH 變數 !!!）
    label = (raw_df["score"].values > THRESH).astype(int)
    out_df = pd.DataFrame({"acct": acct_list, "label": label})
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[inference] thresholded labels saved: {args.out_csv} (threshold={THRESH})")

    # >>> 新增：顯示轉換成 1 的數量 <<<
    num_pos = int(label.sum())
    total   = len(label)
    print(f"[inference] total positive (label=1): {num_pos} / {total}")

if __name__ == "__main__":
    main()
