import os
import random
import mysql.connector
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression      #
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops  #
from skimage.color import rgb2gray
from collections import defaultdict
from scipy.spatial.distance import pdist



# 从MySQL表格里取数据标签及路径
def data_from_table(host="localhost", user="root", password="123456", database="pmsm_fault", table_name="motor1"):
    # root = '../dataset'
    all_data = []
    try:
        connection = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cursor = connection.cursor()
        query = f"SELECT class, sub_class, magnitude, file_path, file_name FROM {table_name}"
        cursor.execute(query)
        rows = cursor.fetchall()
        group_mags = defaultdict(set)  # 收集每个(class, sub_class)下的所有magnitude
        for row in rows:
            class_name, sub_class, magnitude, *_ = row
            key = (class_name, sub_class)
            group_mags[key].add(magnitude)
        mag_rank_map = {}  # magnitude映射
        for key, mag_set in group_mags.items():
            sorted_mags = sorted(mag_set)
            mag_rank_map[key] = {mag: i + 1 for i, mag in enumerate(sorted_mags)}

        all_data = []
        for row in rows:
            class_name, sub_class, magnitude, file_path, file_name = row
            key = (class_name, sub_class)
            mag_rank = mag_rank_map[key][magnitude]
            label_parts = [class_name]
            if sub_class:
                label_parts.append(sub_class)
            label_parts.append(str(mag_rank))
            label = '_'.join(label_parts)
            full_path = os.path.join(file_path, file_name)
            all_data.append((full_path, label, class_name, sub_class, mag_rank))
        random.shuffle(all_data)
        return all_data

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()


# 箱线图
def box_by_subclass_magnitude(feats, sub_labels, mag_labels, palette_dict, metric="euclidean",title="Distance to sub-class centroid"):
    feats = np.asarray(feats)
    subs = np.asarray(sub_labels)
    mags = np.asarray(mag_labels)

    # centroids = {}           # 每个 sub-class 的质心
    # for sc in np.unique(subs):
    #     centroids[sc] = feats[subs == sc].mean(axis=0)
    # dists = np.linalg.norm(feats - np.stack([centroids[s] for s in subs]), axis=1)
    # df = pd.DataFrame({"SubClass": subs, "Magnitude": mags, "Distance": dists})

    records = []
    for sc in np.unique(subs):          #  计算pairwise距离
        for mg in np.unique(mags):
            mask = (subs == sc) & (mags == mg)
            idx  = np.where(mask)[0]
            if len(idx) < 2:            # 组内不足 2 个样本，无距离可算
                continue
            dvec = pdist(feats[idx], metric=metric)   # 1D vector
            records.extend([(sc, mg, d) for d in dvec])
    df = pd.DataFrame(records, columns=["SubClass", "Magnitude", "Distance"])

    plt.figure(figsize=(1.2 * len(np.unique(subs)) + 2, 5))
    sns.boxplot(data=df, x="SubClass", y="Distance", hue="Magnitude", palette=palette_dict, whis=1.5, showfliers=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# 提取统计特征
def extract_statistical_features(image_path):
    """全局统计 + LBP + GLCM 组合特征"""
    img = Image.open(image_path).convert('RGB')
    img_arr = np.asarray(img) / 255.0

    # -------- 1. 颜色统计 (原来那 12 维) ----------
    mean = np.mean(img_arr, axis=(0, 1))
    var = np.var(img_arr, axis=(0, 1))
    skew = np.mean((img_arr - mean.reshape(1, 1, 3)) ** 3, axis=(0, 1)) / (np.std(img_arr, axis=(0, 1)) ** 3 + 1e-10)
    kurt = np.mean((img_arr - mean.reshape(1, 1, 3)) ** 4, axis=(0, 1)) / (
                np.std(img_arr, axis=(0, 1)) ** 4 + 1e-10) - 3
    stats_feat = np.concatenate([mean, var, skew, kurt])  # 12 维

    # -------- 2. LBP 纹理直方图 ----------
    gray = rgb2gray(img_arr).astype(np.uint8)                      # 取 8bit 灰度方便构建共生矩阵
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')    # uniform LBP → bins = P+2 = 10
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 1), range=(0, 10), density=True)

    # -------- 3. GLCM 统计 ----------
    gray8 = (gray * 255)
    glcm = graycomatrix(gray8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    glcm_feat = np.concatenate([contrast, homogeneity, energy, correlation])  # 4×4 = 16 维

    # -------- 4. 拼接输出 ----------
    return np.concatenate([stats_feat, lbp_hist, glcm_feat])  # 12 + 10 + 16 = 38 维


# 线性分类器
def logistic_baseline(feats, labels, cv=5, random_state=0):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1, solver="lbfgs"))
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, feats, labels, cv=skf, scoring="accuracy")
    print(f"[LogReg baseline] {cv}-fold Acc = {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
    return scores.mean()


# 特征可视化
def visualize_features(data):
    features_list = []
    class_list = []
    sub_class_list = []
    mag_list = []
    for img_path, label, class_name, sub_class, mag_rank in data:
        if os.path.exists(img_path):
            features = extract_statistical_features(img_path)
            features_list.append(features)
            class_list.append(class_name)
            sub_class_list.append(sub_class)
            mag_list.append(mag_rank)

    features_array = np.array(features_list)
    features_scaled = StandardScaler().fit_transform(features_array)
    pca = PCA(n_components=20, random_state=0)  # 先压到 20 维左右
    latent = pca.fit_transform(features_scaled)
    tsne_result = TSNE(n_components=2, perplexity=30, random_state=0, max_iter=1000).fit_transform(latent)

    # 获得唯一类别
    unique_classes = sorted(list(set(class_list)))
    unique_sub_classes = sorted(list(set(sub_class_list)))
    unique_mag = sorted(list(set(mag_list)))

    # 定义颜色（sub_class），形状（class），深浅（mag_rank）
    color_map = pyplot.get_cmap('tab20', len(unique_sub_classes))
    marker_styles = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', 'H', 'd', '|', '_']  # 至少支持十几种
    marker_map = {cls: marker_styles[i % len(marker_styles)] for i, cls in enumerate(unique_classes)}

    # 画点
    plt.figure(figsize=(10, 6))
    for i in range(len(features_list)):
        c = color_map(unique_sub_classes.index(sub_class_list[i]))
        alpha = 0.4 + 0.6 * (mag_list[i] - min(unique_mag)) / max(1, (max(unique_mag) - min(unique_mag)))
        marker = marker_map[class_list[i]]
        plt.scatter(tsne_result[i, 0], tsne_result[i, 1], color=c, alpha=alpha, marker=marker, edgecolors='k',
                    linewidths=0.2, s=60)

    # 图例
    for cls in unique_classes:
        plt.scatter([], [], marker=marker_map[cls], color='grey', label=cls)
    legend1 = plt.legend(title="Class (shape)", loc="upper right", bbox_to_anchor=(1, 1))
    plt.gca().add_artist(legend1)

    # 子类图例（颜色）
    for i, sub in enumerate(unique_sub_classes):
        plt.scatter([], [], color=color_map(i), label=str(sub))
    legend2 = plt.legend(title="class (color)", loc="upper left", bbox_to_anchor=(0, 1))
    plt.gca().add_artist(legend2)
    plt.title('t-SNE Visualization of Statistical Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()

    box_by_subclass_magnitude(features_array, sub_class_list, mag_list, palette_dict=color_map, title="static_feature ")
    logistic_baseline(features_array, sub_class_list)


if __name__ == "__main__":
    all_data = data_from_table()
    visualize_features(all_data)
