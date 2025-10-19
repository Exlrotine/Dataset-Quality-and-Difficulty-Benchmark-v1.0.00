import hashlib
# -*- coding: utf-8 -*-
# 将 RGB 时频图转换为超像素图 (RAG) —— 强化“通道相关性 + 方向各向异性”特征
import os, json, hashlib
import numpy as np
import torch
import dgl
from PIL import Image
from scipy.ndimage import sobel
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.measure import regionprops_table

EPS = 1e-8

def _rag_edges_from_segments(segments):
    a = segments
    pairs_h = np.stack([a[:, :-1].ravel(), a[:, 1:].ravel()], axis=1)
    pairs_v = np.stack([a[:-1, :].ravel(), a[1:, :].ravel()], axis=1)
    pairs = np.concatenate([pairs_h, pairs_v], axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)  # 无向去重
    return pairs


_CACHE_VER = "v2-feat21-slic"

def _knn_edges_from_centroids(cx, cy, k=4):
    # 朴素 KNN：节点数几十个，O(n^2) 也很快
    coords = np.stack([cx, cy], axis=1)  # (N, 2)
    N = coords.shape[0]
    dist2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dist2, np.inf)
    nbrs = np.argpartition(dist2, kth=np.minimum(k, N-1), axis=1)[:, :k]
    src = np.repeat(np.arange(N), nbrs.shape[1])
    dst = nbrs.ravel()
    edges = np.stack([src, dst], axis=1)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return edges

def _cache_key(path, n_segments, add_knn, knn_k):
    st = os.stat(path)
    meta = {
        "path": os.path.abspath(path),
        "mtime": int(st.st_mtime),     # 图片修改时间
        "size": int(st.st_size),       # 图片大小
        "n_segments": int(n_segments),
        "add_knn": bool(add_knn),
        "knn_k": int(knn_k),
        "ver": _CACHE_VER,
    }
    key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()
    return key

def img_to_graphs(
    path, label, device='cpu',
    n_segments=40, add_knn=True, knn_k=4,
    augment=False,                # 训练时 True；验证/测试 False
    log_compress=False, log_beta=4.0,
    per_image_whiten=False,
    aug_prob=0.5,                 # 执行增强的概率
    max_time_shift=8,             # 像素
    max_freq_shift=4,             # 像素
    mask_time_prob=0.25, mask_time_width=0.08,  # 宽度占图宽比例
    mask_freq_prob=0.25, mask_freq_height=0.06, # 高度占图高比例
    ):
    # 1) 读图到 [0,1]
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    h, w, _ = image.shape

    # ---- 预处理：压动态范围 + 训练期增强 +（可选）每图标准化 ----
    if log_compress:
        # 压动态范围：更看重相对差异而非绝对亮度
        image = np.log1p(image * log_beta) / np.log1p(log_beta)

    if augment and np.random.rand() < aug_prob:
        # 时间轴（宽度）平移
        if max_time_shift > 0:
            dx = np.random.randint(-max_time_shift, max_time_shift + 1)
            if dx != 0:
                image = np.roll(image, dx, axis=1)
        # 频带（高度）微移
        if max_freq_shift > 0:
            dy = np.random.randint(-max_freq_shift, max_freq_shift + 1)
            if dy != 0:
                image = np.roll(image, dy, axis=0)
        # 时间遮挡
        if np.random.rand() < mask_time_prob:
            mw = max(1, int(w * mask_time_width))
            x0 = np.random.randint(0, max(1, w - mw + 1))
            image[:, x0:x0+mw, :] = 0.0
        # 频带遮挡
        if np.random.rand() < mask_freq_prob:
            mh = max(1, int(h * mask_freq_height))
            y0 = np.random.randint(0, max(1, h - mh + 1))
            image[y0:y0+mh, :, :] = 0.0

    if per_image_whiten:
        # 每图标准化（按通道）；更稳的做法是全数据集 z-score 放到 trainer 里
        mu = image.reshape(-1, 3).mean(axis=0, keepdims=True)
        sig = image.reshape(-1, 3).std(axis=0, keepdims=True) + 1e-6
        image = (image - mu) / sig
        image = np.clip(image, -3.0, 3.0)           # 为了后续特征稳定，可以把范围裁一下
    gray = rgb2gray(image)

    # 2) 超像素分割（从0开始编号）
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0, channel_axis=-1)
    K = int(segments.max()) + 1

    # 3) 统计聚合：用 bincount 把像素聚到超像素
    idx = segments.ravel()
    counts = np.bincount(idx, minlength=K).astype(np.float32).clip(min=1.0)

    # RGB 通道：mean / std / energy
    feats_mean = []
    feats_std  = []
    feats_msq  = []   # mean of squared (energy)
    sums = []
    sums_sq = []
    for c in range(3):
        x = image[..., c].ravel()
        s  = np.bincount(idx, weights=x, minlength=K).astype(np.float32)
        s2 = np.bincount(idx, weights=x*x, minlength=K).astype(np.float32)
        m = s / counts
        v = np.maximum(s2 / counts - m * m, 0.0)
        feats_mean.append(m)
        feats_std.append(np.sqrt(v + EPS))
        feats_msq.append(s2 / counts)
        sums.append(s); sums_sq.append(s2)

    mean_r, mean_g, mean_b = feats_mean
    std_r, std_g, std_b    = feats_std
    msq_r, msq_g, msq_b    = feats_msq

    # 3.1 通道相关性（Pearson r）
    def cov_from_sums(sum_x, sum_y, sum_xy):
        ex  = sum_x / counts
        ey  = sum_y / counts
        exy = sum_xy / counts
        return exy - ex * ey

    sum_rg = np.bincount(idx, weights=(image[...,0]*image[...,1]).ravel(), minlength=K).astype(np.float32)
    sum_gb = np.bincount(idx, weights=(image[...,1]*image[...,2]).ravel(), minlength=K).astype(np.float32)
    sum_rb = np.bincount(idx, weights=(image[...,0]*image[...,2]).ravel(), minlength=K).astype(np.float32)
    cov_rg = cov_from_sums(sums[0], sums[1], sum_rg)
    cov_gb = cov_from_sums(sums[1], sums[2], sum_gb)
    cov_rb = cov_from_sums(sums[0], sums[2], sum_rb)
    r_rg = cov_rg / (std_r * std_g + EPS)
    r_gb = cov_gb / (std_g * std_b + EPS)
    r_rb = cov_rb / (std_r * std_b + EPS)

    # 3.2 方向各向异性（结构张量） + 水平/垂直梯度
    gx = sobel(gray, axis=1).astype(np.float32)  # x: 水平变化（列方向）
    gy = sobel(gray, axis=0).astype(np.float32)  # y: 垂直变化（行方向）
    abs_gx = np.abs(gx).ravel()
    abs_gy = np.abs(gy).ravel()
    mean_abs_gx = np.bincount(idx, weights=abs_gx, minlength=K).astype(np.float32) / counts
    mean_abs_gy = np.bincount(idx, weights=abs_gy, minlength=K).astype(np.float32) / counts

    Sxx = np.bincount(idx, weights=(gx*gx).ravel(), minlength=K).astype(np.float32)
    Syy = np.bincount(idx, weights=(gy*gy).ravel(), minlength=K).astype(np.float32)
    Sxy = np.bincount(idx, weights=(gx*gy).ravel(), minlength=K).astype(np.float32)

    denom = (Sxx + Syy + EPS)
    cos2theta = (Sxx - Syy) / denom
    sin2theta = (2.0 * Sxy) / denom
    coherence = np.sqrt((Sxx - Syy)**2 + (2.0*Sxy)**2) / denom  # 0~1，越大越“有方向”

    # 3.3 质心与面积
    props = regionprops_table(segments + 1, properties=('centroid', 'area'))
    cy = (props['centroid-0'] / h).astype(np.float32)
    cx = (props['centroid-1'] / w).astype(np.float32)
    area = (props['area'] / (h * w)).astype(np.float32)

    # 3.4 能量占比（3个通道之和为1，提供通道“配比”）
    total_msq = msq_r + msq_g + msq_b + EPS
    ratio_r = msq_r / total_msq
    ratio_g = msq_g / total_msq
    ratio_b = msq_b / total_msq

    # 4) 打包节点特征 (共 20 维)
    node_features = np.stack([
        mean_r, mean_g, mean_b,      # 强度统计
        std_r,  std_g,  std_b,
        msq_r,  msq_g,  msq_b,
        r_rg,   r_gb,   r_rb,         # 通道相关性
        mean_abs_gx, mean_abs_gy,     # 梯度/方向
        coherence, cos2theta, sin2theta,
        cx, cy, area                  # 位置/大小
    ], axis=1).astype(np.float32)

    # 5) 边：RAG 邻接 + 可选 KNN（增强全局连通）
    edges = _rag_edges_from_segments(segments)
    if add_knn and K > 1:
        knn = _knn_edges_from_centroids(cx, cy, k=knn_k)
        edges = np.concatenate([edges, knn], axis=0)
        edges = np.unique(np.sort(edges, axis=1), axis=0)

    src = torch.from_numpy(np.concatenate([edges[:,0], edges[:,1]])).long()
    dst = torch.from_numpy(np.concatenate([edges[:,1], edges[:,0]])).long()

    # 6) 构图（建议在 CPU 上构图；采样在 CPU 更稳）
    g = dgl.graph((src, dst), num_nodes=K, device='cpu')
    g.ndata['feat'] = torch.from_numpy(node_features)  # CPU tensor
    g.ndata['label'] = torch.full((K,), int(label), dtype=torch.long)  # 每个节点同一个类别

    try:  sample_id = int(os.path.basename(path).split('.')[0])
    except Exception:
        sample_id = -1
    g.ndata['sample_id'] = torch.full((K,), sample_id, dtype=torch.long)

    # 如需自环：g = dgl.add_self_loop(g)
    if str(device) != 'cpu':
        g = g.to(device)
    return g



def img_to_graphs_cached(path, label, cache_dir=None, device="cpu", **kwargs):
    # 1) 设定缓存目录（默认在图片所在目录下建一个 _graph_cache）
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(path), "_graph_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 2) 生成 key 与文件名（只根据图片&参数；不含设备/标签）
    key = _cache_key(path,
                     kwargs.get("n_segments", 32),
                     kwargs.get("add_knn", True),
                     kwargs.get("knn_k", 4))
    cache_path = os.path.join(cache_dir, f"{key}.dgl")

    g = None
    if os.path.exists(cache_path):
        try: g = dgl.load_graphs(cache_path)[0][0]  # 只存一张图
        except Exception:            g = None  # 损坏就重新生成

    # 3) 不在缓存里就现构图，并写入缓存（统一存 CPU 图；不存 label）
    if g is None:
        g = img_to_graphs(path, label, device="cpu", **kwargs)
        if "label" in g.ndata:
            g = g.clone()
            g.ndata.pop("label")
        dgl.save_graphs(cache_path, [g])

    # 4) 按本次标签补回 label，并转到目标设备
    g = g.to("cpu")
    g.ndata["label"] = torch.full((g.num_nodes(),), int(label), dtype=torch.long)
    if str(device) != "cpu":
        g = g.to(device)
    return g
