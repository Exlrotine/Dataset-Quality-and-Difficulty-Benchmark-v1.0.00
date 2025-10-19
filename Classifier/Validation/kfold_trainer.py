# kfold_trainer.py
import argparse, json, numpy as np, os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
# from dataloader_cifar import build_full_dataset
from data_loader import build_full_dataset
from trainer import Trainer           # 直接复用你现有 Trainer


def run_kfold(cfg_file: str, k: int = 5, seed: int = 42):
    # 1) 读取 config，拿到 transform / batch_size 等参数
    cfg_path = Path(".config") / cfg_file
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    bs = cfg["batch_size"]
    seed = cfg.get("seed", seed)

    CSV_PATH = "result/k_fold/paircv_k10_round6.csv"      # 用于提取数据子集2-A
    dt = pd.read_csv(CSV_PATH)
    files_in_name = set(dt["filename"].astype(str).str.zfill(5))            # 要筛选的文件范围得是表里有的

    # 2) 构建完整数据集
    data_ts, all_data = build_full_dataset(random_label=cfg.get("random_label", False),
                                  shuffle_ratio=cfg.get("shuffle_ratio", 0), seed=seed)

    # 只筛选文件里的数据集时用
    # keep_indices,new_filenames,new_labels = [], [], []
    # for idx in range(len(all_data)):            # 匹配文件
    #     path, _ = all_data[idx]                 # 取出路径
    #     _, labels = data_ts[idx]
    #     fname = os.path.splitext(os.path.basename(path))[0]  # "00023"
    #     if fname in files_in_name:
    #         keep_indices.append(idx)                       # 标记需要保留的文件名
    #         new_filenames.append(fname)
    #         new_labels.append(labels)
    # df = pd.DataFrame(new_filenames, columns=["fname"])
    # data_ts = Subset(data_ts, keep_indices)                # 得到筛选后的数据集
    # labels = new_labels

    # 全部数据集时用
    filenames = []
    for idx in range(len(all_data)):
        path, _ = all_data[idx]
        fname = os.path.splitext(os.path.basename(path))[0]
        filenames.append(fname)
    df = pd.DataFrame(filenames, columns=["fname"])
    # 3) 用标签做 StratifedKFold，保证每折类别分布一致
    labels = [lbl for _, lbl in all_data]            # full_ds[i] -> (img, label_idx)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n=== Fold {fold}/{k} | train={len(tr_idx)} test={len(te_idx)} ===")
        test_flag  = np.zeros(len(data_ts), dtype=int)
        test_flag[te_idx]  = 1
        df[f"fold{fold}_test"]  = test_flag

        # 3.1 拆成 Subset，再包 DataLoader
        tr_loader = DataLoader(Subset(data_ts, tr_idx), batch_size=bs, shuffle=True)
        te_loader = DataLoader(Subset(data_ts, te_idx), batch_size=bs, shuffle=False)

        # 3.2 把两个 loader 注入 Trainer 实例
        trainer = Trainer(cfg_file, current_fold=fold)          # 用原来的初始化
        trainer.train_loader = tr_loader     # 覆写
        trainer.val_loader   = te_loader

        trainer.fit()                        # 跑一折
    out_dir = Path("result") / cfg["name"]; out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir / f"kfold_KNN_k{k}_all.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=["text.json"], type=str, help="Path to config JSON")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    print(f"Starting K-Fold Cross-Validation with config: {args.cfg}, k_folds: {args.k}")
    run_kfold(cfg_file=args.cfg, k=args.k)
    print("K-Fold Cross-Validation completed.")
