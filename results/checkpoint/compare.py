
"""
从多个 .pkl 文件里抽取 train/val loss，在一个图中画上下两个子图。
前三条曲线颜色固定为：红、黄、蓝。
命令示例：
python compare.py h_dim/h_dim120/h_dim120.pkl h_dim/h_dim192/h_dim192.pkl h_dim/h_dim240/h_dim240.pkl --outdir h_dim --title "H_dim comparison"
"""

import argparse
import os
import pickle
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")  # 关键：无界面环境使用Agg
import matplotlib.pyplot as plt

LOSS_KEYS_TRAIN = [
    "train_loss", "training_loss", "train_losses", "loss_train", "loss_tr", "train",
    "train/total_loss", "loss", "total_loss",
]
LOSS_KEYS_VAL = [
    "val_loss", "valid_loss", "validation_loss", "val_losses", "loss_val", "valid",
    "eval_loss", "dev_loss", "val/total_loss",
]

def _is_number(x):
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False

def _flatten_dicts(d):
    if isinstance(d, dict):
        yield d
        for v in d.values():
            yield from _flatten_dicts(v)
    elif isinstance(d, (list, tuple)):
        for v in d:
            yield from _flatten_dicts(v)

def _extract_sequence_from_list(lst, key_priority):
    if not isinstance(lst, (list, tuple)) or len(lst) == 0:
        return None
    if all(_is_number(x) for x in lst):
        return list(map(float, lst))
    if all(isinstance(x, dict) for x in lst):
        for k in key_priority:
            vals = [x.get(k) for x in lst]
            if all(_is_number(v) for v in vals):
                return list(map(float, vals))
        for k in ["loss", "value"]:
            vals = [x.get(k) for x in lst]
            if all(_is_number(v) for v in vals):
                return list(map(float, vals))
    if all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in lst):
        second = [x[1] for x in lst]
        if all(_is_number(v) for v in second):
            return list(map(float, second))
    return None

def _extract_from_dict(d, key_priority):
    for k in key_priority:
        if k in d:
            seq = d[k]
            if isinstance(seq, (list, tuple)):
                seq2 = _extract_sequence_from_list(seq, key_priority)
                if seq2 is not None:
                    return seq2
            elif _is_number(seq):
                return [float(seq)]
    for k in ["history", "logs", "metrics", "stats", "train_history", "val_history"]:
        if k in d:
            seq = _try_extract_series(d[k], key_priority)
            if seq is not None:
                return seq
    return None

def _try_extract_series(obj, key_priority):
    if isinstance(obj, (list, tuple)):
        return _extract_sequence_from_list(obj, key_priority)
    if isinstance(obj, dict):
        seq = _extract_from_dict(obj, key_priority)
        if seq is not None:
            return seq
    for sub in _flatten_dicts(obj):
        seq = _extract_from_dict(sub, key_priority)
        if seq is not None:
            return seq
    return None

def load_history(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    train_seq = _try_extract_series(data, LOSS_KEYS_TRAIN)
    val_seq   = _try_extract_series(data, LOSS_KEYS_VAL)
    return train_seq, val_seq

def pad_to_same_length(series_list):
    max_len = max((len(s) for s in series_list if s is not None), default=0)
    out = []
    for s in series_list:
        if s is None:
            out.append([None] * max_len)
        else:
            padded = list(s) + [None] * (max_len - len(s))
            out.append(padded)
    return out, max_len

def main():
    parser = argparse.ArgumentParser(description="在一个图里画 train/val loss（上下两个子图）。")
    parser.add_argument("files", nargs="+", help=".pkl 文件路径")
    parser.add_argument("--outdir", default="plots", help="输出目录")
    parser.add_argument("--title", default="Loss Curves Comparison", help="图标题")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    labels = [Path(f).stem for f in args.files]

    train_series, val_series = [], []
    for f in args.files:
        tr, va = load_history(f)
        train_series.append(tr)
        val_series.append(va)

    train_padded, T = pad_to_same_length(train_series)
    val_padded, V   = pad_to_same_length(val_series)
    epochs, epochs_v = list(range(T)), list(range(V))

    # 颜色：前三个 红、绿、蓝，更多的走默认色循环
    ordered_colors = ["red", "green", "blue"]
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    def color_for_index(i):
        if i < len(ordered_colors):
            return ordered_colors[i]
        return default_cycle[(i - len(ordered_colors)) % max(1, len(default_cycle))]

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2, 1, 1)
    for i, (s, lab) in enumerate(zip(train_padded, labels)):
        if any(v is not None for v in s):
            ax1.plot(epochs, s, label=f"{lab} train", color=color_for_index(i))
    ax1.set_title(f"{args.title} - Training & Validation")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    for i, (s, lab) in enumerate(zip(val_padded, labels)):
        if any(v is not None for v in s):
            ax2.plot(epochs_v, s, label=f"{lab} val", color=color_for_index(i))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(args.outdir, "combined_loss_plot.png")
    plt.savefig(out_path, dpi=160)
    print("Saved:", out_path)  # 明确打印路径

if __name__ == "__main__":
    main()
