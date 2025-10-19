# kpair_cv.py  ---------------------------------------------------------------
import argparse, json, itertools, numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
import torch, torch.nn.functional as F
import pandas as pd
from collections import Counter
from data_loader import build_full_dataset          # 和你现有保持一致
from trainer import Trainer                         # 复用、不改 trainer


# ---------- 主函数 ----------
def run_paircv(cfg_file: str, k: int = 3, seed: int = 42):

    # 1. 读取配置 ----------------------------------------------------------------
    cfg = json.load(open(Path(".config")/args.cfg, encoding="utf-8"))
    bs   = cfg["batch_size"]
    seed = cfg.get("seed", seed)

    CSV_PATH = "result/k_fold/paircv_k10_round4.csv"  # 你的结果文件
    K_MINUS_1 = 9                          #  一行里 prob_0 … prob_8
    prob_THRESH = 0.5  # 60 %
    df = pd.read_csv(CSV_PATH)
    prob_cols = [f"prob_{i}" for i in range(K_MINUS_1)]          # 9次概率
    correct_cols =  [f"correct_{i}" for i in range(K_MINUS_1)]    # 9次是否命中
    loss_cols =  [f"loss_{i}" for i in range(K_MINUS_1)]          # 9次损失
    condition1 = (df[prob_cols] < prob_THRESH).all(axis=1) # | (df[loss_cols].values.sum(axis=1)<2)
    condition2 = (df[prob_cols].values * df[correct_cols].values).sum(axis=1) < 0.3
    condition3 = (df[loss_cols].values.sum(axis=1) > 36) # | (df[loss_cols] > 1.8).all(axis=1)
    # condition3 = (df[loss_cols] > 1.2).all(axis=1)
    # drop_mask = condition1 | condition2 | condition3   # 取或，满足一个条件就被掩掉
    drop_mask =  condition1| condition2 | condition3
    files_to_drop = set(df.loc[drop_mask, "filename"].astype(str).str.zfill(5))   # 找到需要丢弃的文件名
    files_in_name = set(df["filename"].astype(str).str.zfill(5))            # 要筛选的文件范围得是表里有的

    # 2. 构建完整数据集 ----------------------------------------------------------
    data_ts, all_data = build_full_dataset(random_label=cfg.get("random_label", False),
                                 shuffle_ratio=cfg.get("shuffle_ratio", 0), seed=seed)
    keep_indices,new_filenames,new_labels = [], [], []
    for idx in range(len(all_data)):            # 匹配文件
        path, _ = all_data[idx]                 # 取出路径
        _, labels = data_ts[idx]
        fname = os.path.splitext(os.path.basename(path))[0]  # "00023"
        # if (labels == 4  or fname not in files_to_drop) and fname in files_in_name:
        if (fname not in files_to_drop) and fname in files_in_name:
            keep_indices.append(idx)                       # 标记需要保留的文件名
            new_filenames.append(fname)
            new_labels.append(labels)
    data_ts = Subset(data_ts, keep_indices)                # 得到清洗后的数据集
    # labels    = [lbl for _, lbl in data_ts]              # 从清洗后的数据集得到标签
    labels = new_labels
    print(f"keep {len(keep_indices)} / {len(all_data)} samples")
    N = len(data_ts)

    cls_cnt = Counter(labels)           # 计数文件名
    print("Samples per class after filter:")
    for c in sorted(cls_cnt):
        print(f"  Class {c:>2}: {cls_cnt[c]}")

    # 3. k 折划分 ----------------------------------------------------------------
    skf    = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds  = [test_idx for _, test_idx in skf.split(np.zeros(N), labels)]

    # 4. 初始化统计表 -------------------------------------------------------------
    correct_mat = np.zeros((N, k-1), dtype=np.int8)   # 每样本 9 个 correct(0/1)
    prob_mat    = np.zeros((N, k-1), dtype=np.float32)
    loss_mat    = np.zeros((N, k-1), dtype=np.float32)
    appear_cnt  = np.zeros(N,          dtype=np.int8) # 计数确认<=9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5. 跑 C(k,2) 组合 ----------------------------------------------------------
    all_pairs = list(itertools.combinations(range(k), 2))
    for pair_id, (f1, f2) in enumerate(all_pairs, 1):
        test_idx   = np.concatenate([folds[f1], folds[f2]])
        train_idx  = np.concatenate([folds[i] for i in range(k) if i not in (f1, f2)])

        trainer = Trainer(cfg_file, current_fold=pair_id)
        trainer.train_loader = DataLoader(Subset(data_ts, train_idx),
                                          batch_size=bs, shuffle=True, drop_last=False)
        trainer.val_loader   = DataLoader(Subset(data_ts, test_idx),
                                          batch_size=bs, shuffle=False, drop_last=False)
        trainer.fit()                                   # 正常训练

        # 6. 评估本轮测试集 ------------------------------------------------------
        model = trainer.model.to(device).eval()
        for i in range(0, len(test_idx), bs):
            batch_idx = test_idx[i:i+bs]
            imgs  = torch.stack([data_ts[j][0] for j in batch_idx]).to(device)
            y     = torch.tensor([labels[j] for j in batch_idx], device=device)
            with torch.no_grad():
                logits = model(imgs)
            probs = logits.softmax(1)
            preds = probs.argmax(1)
            losses = F.cross_entropy(logits, y, reduction='none')

            # 写入矩阵：该样本第 appear_cnt[j] 列
            for pos, j in enumerate(batch_idx):
                c = appear_cnt[j]                      # 第几次出现
                correct_mat[j, c] = int(preds[pos]==y[pos])
                prob_mat[j, c]    = probs[pos, y[pos]].item()
                loss_mat[j, c]    = losses[pos].item()
                appear_cnt[j]    += 1

        torch.cuda.empty_cache()
        del trainer, model

    # 7. 保存结果 ----------------------------------------------------------------
    assert appear_cnt.max() == k-1, "每个样本应出现 9 次；若 <9 说明 early_stop/采样错误"
    cols = {"filename": new_filenames, "label":   labels, }
    for n in range(k-1):
        cols[f"correct_{n}"] = correct_mat[:, n]
    for n in range(k-1):
        cols[f"prob_{n}"]    = prob_mat[:, n]
    for n in range(k-1):
        cols[f"loss_{n}"]    = loss_mat[:, n]

    df = pd.DataFrame(cols)
    out_dir = Path("result") / cfg["name"]; out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir / f"paircv_k{k}_{prob_THRESH}.csv", index=False)

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="text.json")
    parser.add_argument("--k",  type=int, default=10)
    args = parser.parse_args()

    run_paircv(cfg_file=args.cfg, k=args.k)
