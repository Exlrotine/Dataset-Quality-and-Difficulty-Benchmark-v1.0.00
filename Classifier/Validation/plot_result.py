import glob, argparse, re, sys
import torch
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib  as mpl
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from skdim.id import MLE,TwoNN
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",  required=True, help="实验目录名 (= Trainer.cfg['name'])")
    parser.add_argument("--sample", type=int, default=4680, help="最多抽多少张样本做 t-SNE ")
    parser.add_argument("--perp", type=float, default=40.0, help="t-SNE perplexity")
    parser.add_argument("--model_name", type=str, default="CNNModel", help="模型名称，例如 CNNModel, ResNet18等")
    parser.add_argument("--batch_size", type=int, default=64, help="生成特征时的batch_size")
    parser.add_argument("--mode", required=True, choices=["acc", "f1", "tsne", "cm", "cm_std", "box", "gap"],
                        help="acc=曲线, cm=混淆矩阵, box=箱线图, beta=log拟合斜率, gap=Train‑Val Gap 热图")
    args = parser.parse_args()
    exp_dir   = Path("result") / args.exp
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)
    if args.mode == "acc":       # 画准确率曲线
        metrics = sorted(exp_dir.glob(f"{args.exp}_*.csv"))
        plot_val_metric(metrics, col="val_acc", ylabel = "Validation-Acc",
                        title = f"{args.exp} • Validation-Acc", scale = 100)
    elif args.mode == "f1":      # 画F1分数曲线
        metrics = sorted(exp_dir.glob(f"{args.exp}_*.csv"))
        plot_val_metric(metrics, col="val_f1", ylabel = "F1 Score (%)",
                         title = f"{args.exp} • Validation-F1", scale = 100)
    elif args.mode == "cm":     # 画混淆矩阵
        cms, classes = load_cm_stack(exp_dir / f"cm_{args.exp}_*.csv")
        plot_avg_cm(cms, classes, title=f"{args.exp} • Mean Confusion Matrix")
    elif args.mode == "cm_std":  # 画混淆矩阵
        cms, classes = load_cm_stack(exp_dir / f"cm_{args.exp}_*.csv")
        plot_cm_std(cms, classes)
    elif args.mode == "tsne":  # 画特征图
        feature_file_name = f"m{args.model_name}_batch{args.batch_size}.gap.pt"
        feature_file_path = exp_dir / feature_file_name
        x, y = load_pt(feature_file_path)
        plot_tsne(x, y, args.sample, args.exp, args.perp)

    elif args.mode == "gap":
        metrics = sorted(exp_dir.glob(f"{args.exp}_seed*.csv"))
        plot_gap_heatmap(metrics, title=f"{args.exp} • Train–Val Gap")

    else:  #  box 模式
        paths = sorted(exp_dir.glob(f"{args.exp}_seed*_r*.csv"))
        plot_box_last30(paths, title=f"{args.exp} • Last-30-Epoch Val-Acc")


def load_cm_stack(pattern):
    dfs  = [pd.read_csv(f, index_col=0) for f in glob.glob(str(pattern))]
    cms  = np.stack([df.values for df in dfs])        # (S, C, C)
    cls  = dfs[0].index.tolist()
    return cms, cls


def plot_box_last30(metrics_paths, title="Boxplot"):
    pattern = re.compile(r"_r([0-9.]+)\.csv$")   # 从文件名提取 r
    buckets = defaultdict(list)                  # ratio → list[val_acc]

    for p in metrics_paths:
        ratio = pattern.search(p.name).group(1)  # '0.8' 这样的字符串
        df = pd.read_csv(p)
        vals = df["val_acc"].tail(30).dropna() * 100
        buckets[ratio].extend(vals.values.tolist())
    ratios     = sorted(buckets.keys(), key=float)
    data       = [buckets[r] for r in ratios]
    variances = [np.var(d, ddof=1) for d in data]  # ddof=1 使用样本方差
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, showfliers=False)
    for i, var in enumerate(variances):
        ax.text(i + 1, max([max(d) for d in data]) * 1.02, f'Var: {var:.2f}',
                ha='center', va='bottom', fontsize=8, color='blue')
    ax.set_xlabel("train_ratio")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, axis='y')
    plt.tight_layout(); plt.show()


def plot_avg_cm(cm_stack, classes, title="Average CM", cell_font_size=8):
    mean = cm_stack.mean(0)
    std  = cm_stack.std(0)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(mean, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, values_format=".2f", colorbar=False)
    # Adjust font size of the cell text
    for text in disp.text_.ravel():
        text.set_fontsize(cell_font_size)
    ax.set_title(title + f"\n(n = {cm_stack.shape[0]} runs)")
    plt.tight_layout(); plt.show()


def plot_cm_std(cm_stack, classes):
    std = cm_stack.std(0) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(std, cmap="Reds")
    for (i, j), v in np.ndenumerate(std):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_title("Per-cell Std (%) across seeds")
    fig.colorbar(im, ax=ax, label="Std of recall %")
    plt.tight_layout(); plt.show()


def plot_val_metric(metrics_paths, col, ylabel, title, scale=100):
    """绘制某个验证指标随 epoch 变化的曲线（支持多实验叠加）。"""
    # pat = re.compile(r"_r([0-9.]+)\.csv$")
    pat = re.compile(r"shuffle([0-9.]+)\.csv$")
    # pat = re.compile(r"m([a-zA-Z0-9]+)_r0.8\.csv$")
    # pat = re.compile(r"phi([-a-zA-Z0-9]+)\.csv$")
    cmap = plt.get_cmap("plasma", len(metrics_paths))
    fig, ax = plt.subplots(figsize=(5, 3))
    for idx, p in enumerate(metrics_paths):
        df = pd.read_csv(p)
        if col not in df.columns:
            print(f"[WARN] {p.name} 缺少列 {col}，跳过")
            continue
        y = df[col] * scale          # 0-1 → 百分比
        x = df["epoch"] if "epoch" in df.columns else range(len(y))
        m = pat.search(p.name)
        # alpha_part=  m.group(1)
        # b_part = m.group(2)
        # lbl = f"a{alpha_part}_b{b_part}" if m else p.stem # 例如：alphaExp1_bModelA
        lbl = m.group(1) if m else p.stem
        ax.plot(x, y, label=lbl, linewidth=1.5, color=cmap(idx))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    # Move legend outside the plot to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.show()



def load_pt(pt_path):
    """读取 ext_features.py 保存的 {'feat':…, 'label':…}"""
    d = torch.load(pt_path, map_location="cpu")
    # print(len(np.unique(d["label"])))
    feat, lbl = d["feat"], d["label"]
    print(feat.shape)

    mle = MLE()
    d_int = mle.fit_transform(feat)
    print("MLE Intrinsic Dimension =", float(d_int))

    pca = PCA().fit(feat)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    d95 = np.searchsorted(cum_var, 0.95) + 1  # 95% 方差需要多少维
    print("PCA-95% 维度 =", d95)

    d_int = TwoNN().fit_transform(feat)
    print("TwoNN  :", d_int)

    if isinstance(feat, torch.Tensor): feat = feat.numpy()
    if isinstance(lbl,  torch.Tensor): lbl  = lbl.numpy()
    feat = feat.reshape(feat.shape[0], -1)   # (N, D)  无论 raw 还是 gap 都 OK
    return feat, lbl



def plot_tsne(x, y, sample, title="t-SNE", perp=40.):
    if len(x) != len(y):
        n = min(len(x), len(y))
        print(f"[WARN] feat {len(x)} vs label {len(y)} → 裁剪到 {n}")
        x, y = x[:n], y[:n]

    if sample and len(x) > sample:
        idx = np.random.choice(len(x), sample, replace=False)
        x, y = x[idx], y[idx]

    tsne = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", max_iter=1000, random_state=0)
    X_2d = tsne.fit_transform(x)

    classes = np.unique(y)
    n_cls   = len(classes)
    cmap    = ListedColormap(mpl.cm.get_cmap("gist_ncar", n_cls)(range(n_cls)) )

    class2idx = {c:i for i,c in enumerate(classes)}
    colors = [class2idx[c] for c in y]

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(X_2d[:,0], X_2d[:,1],     c=colors, cmap=cmap,        s=8, alpha=0.9)
    ax.set_title(f"{title} · t-SNE  N={len(y)}")
    ax.set_xticks([]); ax.set_yticks([])

    handles = [Line2D([0], [0], marker="o", linestyle="",  markerfacecolor=cmap(i), markersize=6, label=str(cls))
               for cls, i in class2idx.items()]
    ax.legend(handles=handles, title="Class",  bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=7)
    plt.tight_layout(); plt.show()


def plot_gap_heatmap(metrics_paths, title="Gap Heatmap (train‑val)", scale=100):
    """绘制 Train‑Val Gap 热图。"""
    rows = []
    epoch_max = 0
    max_val_accs = {}  # 新增：用于存储每个 run 的最终 val_acc
    _id = {}
    for p in metrics_paths:
        df = pd.read_csv(p)
        if {'train_eval_loss', 'val_acc'}.issubset(df.columns):
            match = re.search(r'_wd([\d\.eE-]+)_dr([\d\.eE-]+)', p.stem)
            if match:  run_id = match.group(2)
            else:      run_id = p.stem
            max_val_accs[run_id] = df['val_acc'].max()  # 存储最高 val_acc
            _id[run_id] = run_id  # 存储run_id
            for _, row in df.iterrows():
                # gap = (row['val_loss'] - row['train_loss'])
                gap = (row['val_loss'] - row['train_eval_loss'])
                rows.append({'run': run_id, 'epoch': int(row['epoch']), 'gap': gap})
                epoch_max = max(epoch_max, int(row['epoch']))

    df_all = pd.DataFrame(rows)
    pivot = df_all.pivot(index='run', columns='epoch', values='gap')
    # 按最终验证准确率排序，让 val_acc 最高的 run 在最前面
    # ordered_runs = sorted(pivot.index, key=lambda r: max_val_accs.get(r, -np.inf),reverse=True)
    ordered_runs = sorted(pivot.index, key=lambda r: _id.get(r, -np.inf), reverse=True)
    pivot = pivot.loc[ordered_runs]

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(ordered_runs) + 2))
    im = ax.imshow(pivot, aspect='auto', interpolation='nearest', cmap='viridis')   # 'viridis'  "plasma"  'hot'
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Train – Val loss (pp)', rotation=270, labelpad=15)

    # 轴标签
    ax.set_yticks(np.arange(len(ordered_runs)))
    ax.set_yticklabels([f"{run} ({max_val_accs[run]*100:.2f}%)" for run in ordered_runs], fontsize=7)
    ax.set_xticks(np.arange(epoch_max + 1))
    ax.set_xticklabels(np.arange(epoch_max + 1), fontsize=7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Run')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

