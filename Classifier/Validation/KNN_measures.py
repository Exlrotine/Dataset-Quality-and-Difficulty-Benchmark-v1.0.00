# oof_extract_min.py

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy import linalg
from statsmodels.stats.proportion import proportion_confint
from scipy.linalg import orthogonal_procrustes
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif  # ← 新增
from scipy.stats import entropy

dataset_names={'all', 'subset2', '2A'}
# dataset_names={'all'}

def main(dataset_name):
    K_list = [30, 60, 90, 120, 150] if dataset_name != '2A' else [10, 20, 30, 40, 50]
    print(f"dataset_name: {dataset_name}")
    data = np.load(f"result/k_fold/features/oof_features_{dataset_name}.npz")

    # -------- 选出 OOF-test --------
    mask = data["is_test"] == 1          # ←换成 0 可以计算训练集
    X, y = data["feat"][mask], data["label"][mask]

    # -------- 新增：互信息 --------
    mi = mutual_info_classif(X, y, discrete_features=False,
                             n_neighbors=3, random_state=0)
    print(f"Mutual information — mean {mi.mean():.4f}, "
          f"median {np.median(mi):.4f}, max {mi.max():.4f}")
    # np.save(f"result/k_fold/features/mi_{dataset_name}.npy", mi)
    # -------- 互信息结束 --------

    num_classes = int(y.max() + 1)
    logk = np.log2(num_classes)

    # ---------- 条件熵 H(Y|X) ----------


    # -------- 下面保持原逻辑 --------
    for k in K_list:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = nbrs.kneighbors(X)
        idx = idx[:, 1:]
        pred = stats.mode(y[idx], axis=1, keepdims=False)[0]
        acc = (pred == y).mean()

        # 统计邻居标签分布
        counts = np.apply_along_axis(lambda v: np.bincount(v, minlength=num_classes), 1, y[idx])  # (N, C)
        probs = counts / k + 1e-12  # Laplace 防 0
        H_Y_given_X = np.mean(entropy(probs.T, base=2))

        # ---------- 总熵 H(Y) ----------
        class_prob = np.bincount(y, minlength=num_classes) / len(y)
        H_Y = entropy(class_prob, base=2)

        # ---------- 互信息 & Fano 下界 ----------
        I_XY = H_Y - H_Y_given_X
        fano_lower = max(0.0, 1.0 - (I_XY + 1.0) / logk)

        lo, hi = proportion_confint(int(acc * len(y)), len(y), method="wilson")
        print(f"k={k:>3}: acc={acc * 100:.2f}%  95%CI=({lo * 100:.2f},{hi * 100:.2f})")
        print(f"I(X;Y) ≈ {I_XY:.3f} bits")
        print(f"Fano lower-bound on error ≥ {fano_lower:.4f}")




def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d['feat'], d['label'], d['fold'], d['fname'], d['is_test']


def plot_tsne(npz_list, labels_list, out_png="tsne.png",
              per_class_alpha=0.6, random_state=0):

    # -------- 收集唯一测试样本 --------
    uniq = {}
    for path, _ in zip(npz_list, labels_list):
        X, y, dom, fnames, is_test = load_npz(path)
        for feat, cls, fname, is_t in zip(X, y, fnames, is_test):
            if is_t == 1:                 # 只取 OOF-test
                uniq[fname] = (feat, cls)
    X, y = map(np.array, zip(*uniq.values()))

    # -------- t-SNE --------
    X_std = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(30, X.shape[0]//3),  init="pca", learning_rate='auto',
                metric='cosine', random_state=random_state)
    emb = tsne.fit_transform(X_std)

    # -------- 绘图：颜色=类别，形状统一 --------
    classes = np.unique(y)
    n_cls   = len(classes)

    # ① 选一个够多颜色的 colormap；tab20 ≤20 色，超过用 hsv
    if n_cls <= 20:    cmap = plt.get_cmap('tab20', n_cls)
    elif n_cls <= 256:   cmap = plt.get_cmap('hsv', n_cls)
    else:  raise ValueError("Too many classes for color map")

    plt.figure(figsize=(4, 3))
    for i, c in enumerate(classes):
        idx = (y == c)
        plt.scatter(emb[idx, 0], emb[idx, 1], s=18, marker='o', color=cmap(i),alpha=per_class_alpha, )# label=f"class {c}")

    plt.title("t-SNE of features (test only)")
    # plt.legend(markerscale=1.2, fontsize=8, frameon=False, ncol=4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return emb, y


def fid_like(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance used in FID."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)


def mmd_rbf(x, y, gamma=None):
    """Maximum Mean Discrepancy with RBF kernel."""
    if gamma is None:
        # median heuristic
        med = np.median(pairwise_distances(np.vstack([x,y])))
        gamma = 1. / (2 * med**2 + 1e-9)
    kxx = np.exp(-pairwise_distances(x, x)**2 * gamma).mean()
    kyy = np.exp(-pairwise_distances(y, y)**2 * gamma).mean()
    kxy = np.exp(-pairwise_distances(x, y)**2 * gamma).mean()
    return kxx + kyy - 2*kxy


def metric_distribution(npz_A, npz_B, class_names=None, save_csv=None):
    XA, yA, _, _, tA = load_npz(npz_A)
    XB, yB, _, _, tB = load_npz(npz_B)

    XA, yA = XA[tA==1], yA[tA==1]   # 只 test
    XB, yB = XB[tB==1], yB[tB==1]
    assert XA.shape[1] == XB.shape[1]
    D = XA.shape[1]
    classes = sorted(set(yA.tolist()) & set(yB.tolist()))

    # --------- ① 先做“跨域正交对齐” (B ➜ A) ---------
    ref_mu, trg_mu = [], []
    for c in classes:
        ref_mu.append(XA[yA == c].mean(0))
        trg_mu.append(XB[yB == c].mean(0))
    ref_mu, trg_mu = np.stack(ref_mu), np.stack(trg_mu)
    valid = (~np.isnan(ref_mu).any(1)) & (~np.isnan(trg_mu).any(1))
    if valid.sum() >= 2:  # 至少两类才能求 R
        R, _ = orthogonal_procrustes(trg_mu[valid], ref_mu[valid])
        XB = XB @ R  # ← 对齐
    else:
        print("[metric_distribution]  Warning: not enough "
              "valid classes for domain alignment, skip.")

    # --------- ② 原有度量逻辑保持不变 ----------
    rows = []
    for c in classes:
        Xa = XA[yA == c]; Xb = XB[yB == c]
        mu_a, mu_b = Xa.mean(0), Xb.mean(0)
        Sa = np.cov(Xa.T) if Xa.shape[0] > 1 else np.eye(D)
        Sb = np.cov(Xb.T) if Xb.shape[0] > 1 else np.eye(D)
        cen_l2  = np.linalg.norm(mu_a - mu_b)
        cen_cos = 1 - (mu_a @ mu_b) / (np.linalg.norm(mu_a)* np.linalg.norm(mu_b)+1e-9)
        fidv = fid_like(mu_a, Sa, mu_b, Sb)      # 脚本中已有这个函数
        mmd  = mmd_rbf(Xa, Xb)
        intra_A, intra_B = np.trace(Sa), np.trace(Sb)

        rows.append({
            'class': c if class_names is None else class_names[c],
            'n_A': Xa.shape[0],       'n_B': Xb.shape[0],
            'cen_dist_L2': cen_l2,    'cen_dist_cos': cen_cos,
            'FID_like':    fidv,      'MMD_RBF': mmd,
            'intra_var_A': intra_A,   'intra_var_B': intra_B
        })
    df = pd.DataFrame(rows)
    if save_csv: df.to_csv(save_csv, index=False)
    return df


if __name__ == "__main__":
    for dataset_name in dataset_names:
        main(dataset_name)
        plot_tsne([f"result/k_fold/features/oof_features_{dataset_name}.npz"],
                 [f"{dataset_name}"], out_png=f"result/k_fold/features/tsne_{dataset_name}.png")

    # 1) 度量分布差异
    df_metrics = metric_distribution("result/k_fold/features/oof_features_subset2.npz",
                                     "result/k_fold/features/oof_features_2A.npz",
                                     save_csv="result/k_fold/features/subset2_vs_2A_metrics.csv")
    print(df_metrics.sort_values("FID_like", ascending=False))
