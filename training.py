# -*- coding: utf-8 -*-
"""
training.py
- 從 ./output/training_data/random_data.csv 讀取交易級資料（由 training_dataset.py 產生）
- 用 ./dataset/acct_alert.csv 動態打標籤（row-level）：
    label = 1 if (from_acct in alert_set) or (to_acct in alert_set) else 0
- 自動分辨數值/類別欄；類別欄（含 from_acct, to_acct）做穩定雜湊 -> embedding
- 訓練 MLP，紀錄 acc / loss / f1_macro / top-5，保存最佳模型
- 輸出所有成果到 ./output/model_run/
- 可當 CLI 執行，也可由 main.py 呼叫 train_entry(**kwargs)
"""

import os
import json
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import hashlib

from model import MLPClassifier

# ------------------ Utils ------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def stable_bucket(s: str, bucket: int, salt: str = "") -> int:
    """
    穩定雜湊（跨程序一致）：md5(value_with_salt) % bucket
    """
    key = (salt + "||" + s).encode("utf-8", errors="ignore")
    return int(hashlib.md5(key).hexdigest(), 16) % bucket

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def topk_acc_from_logits(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    k = min(k, logits.shape[1])
    _, topk = logits.topk(k, dim=1)
    match = (topk == y.unsqueeze(1)).any(dim=1).float().mean().item()
    return match

# ------------------ Dataset ------------------

class TxnDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 label_col: str,
                 numeric_cols: List[str],
                 cat_cols: List[str],
                 cat_buckets: Dict[str, int],
                 hash_salt: str = "col_name::value"):
        self.y = df[label_col].astype(int).values
        self.numeric = df[numeric_cols].astype(float).values if numeric_cols else np.zeros((len(df),0), dtype=float)
        self.cat_cols = cat_cols
        self.cat_buckets = cat_buckets
        self.hash_salt = hash_salt
        self.cat_idx = {}
        for c in cat_cols:
            v = df[c].astype(str).fillna("nan").values
            self.cat_idx[c] = np.array([stable_bucket(x, bucket=cat_buckets[c], salt=c) for x in v], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x_num = torch.tensor(self.numeric[idx], dtype=torch.float32)
        x_cat = {c: torch.tensor(self.cat_idx[c][idx], dtype=torch.long) for c in self.cat_cols}
        y = int(self.y[idx])
        return x_num, x_cat, y

# ------------------ Core ------------------

def prepare_dataframe(input_csv: str,
                      alert_csv: str,
                      from_col: str = "from_acct",
                      to_col: str = "to_acct") -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"input_csv not found: {input_csv}")
    if not os.path.exists(alert_csv):
        raise FileNotFoundError(f"alert_csv not found: {alert_csv}")

    df = pd.read_csv(input_csv)
    missing = [c for c in (from_col, to_col) if c not in df.columns]
    if missing:
        raise ValueError(f"random_data.csv must contain columns: {missing}")

    alert_set = set(pd.read_csv(alert_csv, usecols=["acct"])["acct"].astype(str))
    df[from_col] = df[from_col].astype(str)
    df[to_col] = df[to_col].astype(str)
    df["label"] = (df[from_col].isin(alert_set) | df[to_col].isin(alert_set)).astype(int)
    return df

def build_feature_spec(df: pd.DataFrame,
                       label_col: str,
                       force_cat_cols: List[str] = ["from_acct", "to_acct"],
                       cat_bucket_size: int = 50000) -> Dict:
    feats = [c for c in df.columns if c != label_col]
    numeric_cols, cat_cols = [], []
    for c in feats:
        if is_numeric_series(df[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)
    for c in force_cat_cols:
        if c in numeric_cols:
            numeric_cols.remove(c)
        if c not in cat_cols and c in df.columns:
            cat_cols.append(c)

    cat_buckets = {c: int(cat_bucket_size) for c in cat_cols}
    num_classes = int(pd.Series(df[label_col]).nunique())
    return {
        "label_col": label_col,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "cat_buckets": cat_buckets,
        "num_classes": num_classes,
        "hash_salt": "col_name::value"
    }

def split_train_val(df: pd.DataFrame, label_col: str, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    y = df[label_col]
    tr, va = train_test_split(df, test_size=val_ratio, random_state=seed, stratify=y)
    return tr, va

def _plot_and_save(xs, y_tr, y_va, title, ylabel, out_png):
    plt.figure()
    plt.plot(xs, y_tr, label="train")
    plt.plot(xs, y_va, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def train_once(spec: Dict,
               train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               out_dir: str,
               hidden: List[int],
               lr: float,
               weight_decay: float,
               epochs: int,
               batch_size: int,
               device: str,
               dropout: float,
               cat_emb_dim: int,
               bn: bool) -> str:
    os.makedirs(out_dir, exist_ok=True)
    label_col = spec["label_col"]
    train_ds = TxnDataset(train_df, label_col, spec["numeric_cols"], spec["cat_cols"], spec["cat_buckets"])
    val_ds   = TxnDataset(val_df,   label_col, spec["numeric_cols"], spec["cat_cols"], spec["cat_buckets"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = MLPClassifier(
        input_num_dim=len(spec["numeric_cols"]),
        cat_buckets=spec["cat_buckets"],
        num_classes=spec["num_classes"],
        hidden=hidden,
        dropout=dropout,
        cat_emb_dim=cat_emb_dim,
        bn=bn
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    # 保存 feature_spec
    with open(os.path.join(out_dir, "feature_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,train_f1_macro,val_loss,val_acc,val_f1_macro,val_topk_acc,lr\n")

    best_f1 = -1.0
    best_path = os.path.join(out_dir, "best_model.pt")

    # for plotting
    epochs_x = []
    tr_loss_list, tr_acc_list, tr_f1_list = [], [], []
    va_loss_list, va_acc_list, va_f1_list = [], [], []

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tr_loss, tr_acc, tr_f1, tr_n = 0.0, 0.0, 0.0, 0
        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device)
            x_cat = {k: v.to(device) for k, v in x_cat.items()}
            y = y.to(device)

            logits = model(x_num, x_cat)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += accuracy_from_logits(logits, y) * bs
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            tr_f1  += f1_score(y.detach().cpu().numpy(), preds, average="macro") * bs
            tr_n   += bs

        scheduler.step()

        tr_loss /= tr_n
        tr_acc  /= tr_n
        tr_f1   /= tr_n

        # validate
        model.eval()
        va_loss, va_acc, va_f1, va_topk, va_n = 0.0, 0.0, 0.0, 0.0, 0
        k = min(5, spec["num_classes"])
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num = x_num.to(device)
                x_cat = {k2: v.to(device) for k2, v in x_cat.items()}
                y = y.to(device)

                logits = model(x_num, x_cat)
                loss = criterion(logits, y)
                bs = y.size(0)

                va_loss += loss.item() * bs
                va_acc  += accuracy_from_logits(logits, y) * bs
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                va_f1  += f1_score(y.detach().cpu().numpy(), preds, average="macro") * bs
                va_topk += topk_acc_from_logits(logits, y, k=k) * bs
                va_n    += bs

        va_loss /= va_n
        va_acc  /= va_n
        va_f1   /= va_n
        va_topk /= va_n

        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc": round(tr_acc, 6),
            "train_f1_macro": round(tr_f1, 6),
            "val_loss": round(va_loss, 6),
            "val_acc": round(va_acc, 6),
            "val_f1_macro": round(va_f1, 6),
            "val_topk_acc": round(va_topk, 6),
            "lr": scheduler.get_last_lr()[0]
        }
        print(row)
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(",".join(str(row[k]) for k in ["epoch","train_loss","train_acc","train_f1_macro",
                                                   "val_loss","val_acc","val_f1_macro","val_topk_acc","lr"]) + "\n")

        # collect for plots
        epochs_x.append(epoch)
        tr_loss_list.append(tr_loss); tr_acc_list.append(tr_acc); tr_f1_list.append(tr_f1)
        va_loss_list.append(va_loss); va_acc_list.append(va_acc); va_f1_list.append(va_f1)

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model_state": model.state_dict(),
                "spec": spec,
                "hparams": {
                    "hidden": hidden,
                    "dropout": dropout,
                    "cat_emb_dim": cat_emb_dim,
                    "bn": bn
                }
            }, best_path)
            print(f"Saved best model to {best_path} (val_f1_macro={best_f1:.6f})")

    # plots
    _plot_and_save(epochs_x, tr_loss_list, va_loss_list, "Loss", "loss", os.path.join(out_dir, "curves_loss.png"))
    _plot_and_save(epochs_x, tr_acc_list,  va_acc_list,  "Accuracy", "accuracy", os.path.join(out_dir, "curves_acc.png"))
    _plot_and_save(epochs_x, tr_f1_list,   va_f1_list,   "F1 Macro", "f1_macro", os.path.join(out_dir, "curves_f1.png"))

    return best_path

# ------------------ Public entry for main.py ------------------

def train_entry(
    input_csv: str = "./output/training_data/random_data.csv",
    alert_csv: str = "./dataset/acct_alert.csv",
    out_dir: str = "./output/model_run",
    val_ratio: float = 0.2,
    seed: int = 42,
    cat_bucket_size: int = 50000,
    hidden: List[int] = [256, 128],
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 10,
    batch_size: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dropout: float = 0.2,
    cat_emb_dim: int = 16,
    bn: bool = True
) -> str:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    df = prepare_dataframe(input_csv, alert_csv)
    spec = build_feature_spec(df, label_col="label", force_cat_cols=["from_acct", "to_acct"], cat_bucket_size=cat_bucket_size)
    tr_df, va_df = split_train_val(df, label_col="label", val_ratio=val_ratio, seed=seed)

    best = train_once(
        spec=spec,
        train_df=tr_df,
        val_df=va_df,
        out_dir=out_dir,
        hidden=hidden,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        dropout=dropout,
        cat_emb_dim=cat_emb_dim,
        bn=bn
    )
    return best

# ------------------ CLI ------------------

def parse_list_of_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Train MLP on random_data.csv with alert-based row labels.")
    ap.add_argument("--input_csv", default="./output/training_data/random_data.csv")
    ap.add_argument("--alert_csv", default="./dataset/acct_alert.csv")
    ap.add_argument("--out_dir", default="./output/model_run")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cat_bucket_size", type=int, default=50000)

    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cat_emb_dim", type=int, default=16)
    ap.add_argument("--bn", action="store_true")

    args = ap.parse_args()

    hidden = parse_list_of_ints(args.hidden)
    best = train_entry(
        input_csv=args.input_csv,
        alert_csv=args.alert_csv,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cat_bucket_size=args.cat_bucket_size,
        hidden=hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        dropout=args.dropout,
        cat_emb_dim=args.cat_emb_dim,
        bn=args.bn
    )
    print(f"[training] done. best model: {best}")

if __name__ == "__main__":
    main()
