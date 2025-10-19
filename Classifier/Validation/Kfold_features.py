# oof_extract_min.py
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ---------- 本项目中的脚本 ----------
from data_loader import build_full_dataset  # 返回 data_ts, all_data
# from dataloader_cifar import build_full_dataset  # 返回 data_ts, all_data
from ALL_Model.CNN_Classifier          import CNNModel
from ALL_Model.ResNet_Classifier       import ResNet18SE
from ALL_Model.MobileNet_Classifier    import MobileNetV3Small
from ALL_Model.EfficientNet_Classifier import EfficientNetB0
from ALL_Model.ViT_Classifier          import VisionTransformer

from scipy.linalg import orthogonal_procrustes


MODEL_REGISTRY = {
    "CNNModel": CNNModel,
    "ResNet18": ResNet18SE,
    "MobileNetV3": MobileNetV3Small,
    "EfficientNet": EfficientNetB0,
    "ViT": VisionTransformer,
}

# 用 hook 抓这一层
TARGET_LAYER = {
    "CNNModel":    lambda m: m.conv3,
    "ResNet18":    lambda m: m.layer4[-1],
    "MobileNetV3": lambda m: m.blocks[-1],
    "EfficientNet":lambda m: m.blocks[-1],
    "ViT":         lambda m: m.norm,
}
dataset_names={'all', 'subset2', '2A'}
# dataset_names={'all'}


@torch.no_grad()
def main(args, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) cfg & model
    cfg = json.load(open(Path(".config")/args.cfg, encoding="utf-8"))
    name = cfg.get("model", "CNNModel")
    keys = cfg.get("model_args", {})
    if dataset_name == 'all': keys["num_classes"] = 26
    model = MODEL_REGISTRY[name](**keys)
    model.to(device).eval()
    exp_name = cfg.get("name", None)
    ckpt_dir =  Path("result") / exp_name

    seed = cfg.get("seed", 42)
    data_ts, all_data = build_full_dataset(random_label=cfg.get("random_label", False),
                                           shuffle_ratio=cfg.get("shuffle_ratio", 0), seed=seed)
    # 2) 提取文件名，用于匹配
    filenames = {}
    for i in range(len(all_data)):
        path, _ = all_data[i]
        fname = os.path.splitext(os.path.basename(path))[0]
        filenames[i] = fname

    # 3) 读折信息
    CSV_PATH = ckpt_dir / f"kfold_KNN_k5_{dataset_name}.csv"  # 前面实验保存的结果文件
    df = pd.read_csv(CSV_PATH)
    all_rows = []
    for col in df.columns:
        if not (col.startswith("fold") and col.endswith("_test")): continue  # 跳过name列
        fold_id = int(col[len("fold"):-len("_test")])
        mask_test = df[col] == 1
        mask_train = df[col] == 0

        # test indices ＝ 该折需要的样本位置
        test_fnames = set(df.loc[mask_test, "fname"].astype(str).str.zfill(5))
        test_indices = [i for i, fn in filenames.items() if fn in test_fnames]
        train_fnames = set(df.loc[mask_train, "fname"].astype(str).str.zfill(5))
        train_indices = [i for i, fn in filenames.items() if fn in train_fnames]
        indices = train_indices + test_indices  # 把 test+train 合并，防止重复
        is_test_flags = ([0] * len(train_indices)) + ([1] * len(test_indices))
        loader = DataLoader(Subset(data_ts, indices), batch_size=cfg.get("batch_size", 64),
                            shuffle=False, num_workers=0)

        ckpt = ckpt_dir / f"mEfficientNet_k{fold_id}_{dataset_name}.pth"  # 权重
        # ckpt = ckpt_dir / f"mResNet18_k{fold_id}_{dataset_name}.pth"  # 权重
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)

        hook_layer = TARGET_LAYER[name](model)
        buf = []
        def hook(_, __, out):
            buf.append(out.detach().cpu())
        handle = hook_layer.register_forward_hook(hook)
        cursor = 0
        for x, y in loader:
            idx_batch = indices[cursor: cursor + len(x)]
            flag_batch = is_test_flags[cursor: cursor + len(x)]
            cursor += len(x)
            _ = model(x.to(device))
            f = buf.pop()
            if f.ndim == 4:
                f = F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)

            # 保存全部
            for vec, lbl, idxi, flag in zip(f.cpu(), y, idx_batch, flag_batch):
                fname = filenames[idxi]
                all_rows.append((fname, int(lbl), fold_id, flag, vec.numpy()))
        handle.remove()
        print(f"[Fold {fold_id}] save {len(indices)} feats "
              f"(train {len(train_indices)} + test {len(test_indices)})")

    # 4) 合并
    names = [r[0] for r in all_rows]
    labels = [r[1] for r in all_rows]
    folds = [r[2] for r in all_rows]
    is_test = [r[3] for r in all_rows]  # 新列
    feats = np.stack([r[4] for r in all_rows])

    # ---- 折间对齐 ----
    train_mask = np.array(is_test) == 0  # 直接用
    feats_aligned = align_by_train_centroid(feats, np.array(labels), np.array(folds),
        train_mask, n_classes=13 if dataset_name != 'all' else 26)

    np.savez(out_dir / f"oof_features_{dataset_name}.npz",
             fname=np.array(names), label=np.array(labels),
             fold=np.array(folds), is_test=np.array(is_test, np.uint8),
             feat=feats_aligned)

    pd.DataFrame({"fname": names, "label": labels, "fold": folds, "is_test": is_test}) \
                 .to_csv(out_dir / f"oof_features_{dataset_name}.csv",index=False, encoding="utf-8-sig")
    print(f"[DONE] total {len(names)} → feats {feats_aligned.shape}")


def align_by_train_centroid(feat, label, fold, train_mask, n_classes=None, ref_fold=None):
    """   skip classes (or folds) that lack train samples.   """
    if n_classes is None:
        n_classes = int(label.max()) + 1
    if ref_fold is None:
        ref_fold = int(np.min(fold))
    folds_unique = np.unique(fold)
    D = feat.shape[1]
    aligned = feat.copy()

    # ---- reference fold centroids (train only) ----
    ref_mu_list = []
    for c in range(n_classes):
        idx = (fold == ref_fold) & train_mask & (label == c)
        if idx.any(): ref_mu_list.append(feat[idx].mean(0))
        else:         ref_mu_list.append(np.full(D, np.nan))
    ref_mu = np.stack(ref_mu_list, 0)        # [C, D]

    for fid in folds_unique:
        if fid == ref_fold: continue      # 跳过name列
        mu_list = []
        for c in range(n_classes):
            idx = (fold == fid) & train_mask & (label == c)
            if idx.any():   mu_list.append(feat[idx].mean(0))
            else:           mu_list.append(np.full(D, np.nan))
        mu = np.stack(mu_list, 0)            # [C, D]

        # ---- keep only rows without NaN on both sides ----
        valid = (~np.isnan(mu).any(1)) & (~np.isnan(ref_mu).any(1))
        if valid.sum() < 2:                  # not enough anchors
            print(f"[align] fold {fid}: only {valid.sum()} valid classes, skip rotation.")
            continue                         # leave this fold as‑is
        R, _ = orthogonal_procrustes(mu[valid], ref_mu[valid])
        aligned[fold == fid] = feat[fold == fid] @ R
    return aligned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out", default="result/k_fold/features")
    args = parser.parse_args()
    for dataset_name in dataset_names:
        main(args, dataset_name)
