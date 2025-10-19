import numpy as np
import pandas as pd
import os
import re
import glob


def compute_s_stab(A, N_R, K):
    """  计算 S_stab(D;K)。   返回:  - S_stab 值，在(0,1]范围内。   """
    # 计算每个r的Var_k[A_k(r)]（沿列轴计算方差，使用样本方差 ddof=1）
    var_k = np.var(A, axis=1, ddof=1)
    numerator = np.sum(var_k)
    averages = np.mean(A, axis=1)
    var_r = np.var(averages, ddof=1)
    denominator = N_R * var_r

    # 处理边缘情况
    if denominator == 0:
        if numerator == 0:
            return 1.0  # 完美稳定
        else:
            return 0.0  # 不稳定（但平均值无变异）
    else:
        return np.exp(-numerator / denominator)


def load_and_process_csvs(directory, N_R=5, K=5, expected_r_values=[0.1, 0.25, 0.5, 0.75, 0.9]):
    # 初始化数组，填充NaN以便检查缺失值
    A = np.full((N_R, K), np.nan)
    r_pattern = re.compile(r"_r([0-9]+\.[0-9]+)\.csv$")
    seed_pattern = re.compile(r"_seed([0-9]+)_")
    files = glob.glob(os.path.join(directory, "seed_ratio_seed*_r*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory} matching pattern 'seed_ratio_seed*_r*.csv'")

    # 收集所有唯一的seed值和r值
    seed_values = set()
    r_values = set()
    for file in files:
        seed_match = seed_pattern.search(file)
        r_match = r_pattern.search(file)
        if seed_match:
            seed_values.add(int(seed_match.group(1)))
        if r_match:
            r_values.add(float(r_match.group(1)))

    # 验证r值是否匹配预期
    r_values = sorted(list(r_values))
    if r_values != sorted(expected_r_values):
        raise ValueError(f"Found r values {r_values}, but expected {expected_r_values}")

    # 取前K个seed值（按数值升序）
    seed_values = sorted(list(seed_values))[:K]
    if len(seed_values) < K:
        raise ValueError(f"Found only {len(seed_values)} unique seeds, but K={K} is required")

    # 创建r和seed到索引的映射
    r_to_index = {r: idx for idx, r in enumerate(expected_r_values)}  # 使用预期r值顺序
    seed_to_index = {seed: idx for idx, seed in enumerate(seed_values)}

    for file in files:
        # 提取r值
        r_match = r_pattern.search(file)
        if not r_match:
            print(f"Warning: Could not extract r from {file}")
            continue
        r = float(r_match.group(1))
        if r not in r_to_index:
            print(f"Warning: r={r} in {file} not in expected r values {expected_r_values}, skipping")
            continue
        r_idx = r_to_index[r]  # 映射到0-based行索引

        # 提取seed值
        seed_match = seed_pattern.search(file)
        if not seed_match:
            print(f"Warning: Could not extract seed from {file}")
            continue
        seed = int(seed_match.group(1))
        if seed not in seed_to_index:
            print(f"Warning: seed={seed} in {file} not in top {K} seeds {seed_values}, skipping")
            continue
        k = seed_to_index[seed]  # 映射到0-based列索引

        # 读取CSV的val_acc列并取最大值
        try:
            df = pd.read_csv(file)
            if 'val_acc' not in df.columns:
                print(f"Warning: 'val_acc' column not found in {file}, skipping")
                continue
            val_acc = df['val_acc'].max()  # 取val_acc列的最大值
            if pd.isna(val_acc):
                print(f"Warning: No valid val_acc values in {file}, skipping")
                continue
            A[r_idx, k] = val_acc
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    # 检查是否所有值都填充
    if np.any(np.isnan(A)):
        missing = np.where(np.isnan(A))
        missing_positions = [(expected_r_values[r], seed_values[k]) for r, k in zip(missing[0], missing[1])]
        raise ValueError(f"Missing values in A at positions (r,seed): {missing_positions}")

    print(f"Using r values: {expected_r_values}")
    print(f"Using seeds: {seed_values}")
    return A


def compute_s_bal(M_cc_avg, C):
    """    计算 S_bal(D) 使用平均 M_cc 值。  """
    mu = np.mean(M_cc_avg)  # 平均召回率
    numerator = np.sqrt(np.sum((M_cc_avg - mu) ** 2))  # 偏差的 L2 范数
    denominator = mu  * np.sqrt( C)  # 归一化因子
    if denominator == 0:
        return 0.0  # 避免除以零
    return 1 - (numerator / denominator)


def load_confusion_matrices(directory, C=26, K=5):
    """
    从指定目录读取混淆矩阵文件，提取对角元素并计算平均值。

    参数:
    - directory: 包含混淆矩阵的目录路径。
    - C: 类别数（默认26）。
    - K: 混淆矩阵文件数（默认5）。

    返回:
    - M_cc_avg: 长度为 C 的数组，表示平均对角元素。
    """
    # 获取所有匹配的文件
    files = glob.glob(os.path.join(directory, "cm_confusion_matrix_seed*_r*.csv"))

    if len(files) != K:
        raise ValueError(f"Expected {K} files starting with 'cm_confusion_matrix_seed*_r*.csv', found {len(files)}")

    # 初始化存储对角元素的列表
    M_cc_all = []

    for file in files:
        try:
            # 读取 CSV 文件，跳过第一行和第一列
            df = pd.read_csv(file, index_col=0).iloc[:, 0:]  # 跳过第一列作为索引，第一行作为列名
            if df.shape != (C, C):
                raise ValueError(f"Matrix in {file} should be {C}x{C} after skipping headers, got {df.shape}")
            # 提取对角元素
            M_cc = np.diag(df.to_numpy())
            M_cc_all.append(M_cc)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not M_cc_all:
        raise ValueError("No valid confusion matrices loaded")

    if len(M_cc_all) != K:
        print(f"Warning: Only {len(M_cc_all)} out of {K} files processed successfully")

    # 计算所有文件的 M_cc 平均值
    M_cc_avg = np.mean(M_cc_all, axis=0)
    return M_cc_avg

def main():
    # 参数
    directory = "result\\seed_ratio"  # CSV文件所在目录
    N_R = 5
    K = 5
    expected_r_values = [0.1, 0.25, 0.5, 0.75, 0.9]  # 预期的r值

    try:
        # 读取并处理CSV文件
        A = load_and_process_csvs(directory, N_R, K, expected_r_values)
        print("Loaded A matrix:")
        print(A)

        # 计算S_stab
        s = compute_s_stab(A, N_R, K)
        print(f"S_stab(D;K) = {s}")

    except Exception as e:
        print(f"Error: {e}")


def main2():
    # 参数
    directory = "result\\confusion_matrix"  # 混淆矩阵文件所在目录
    C = 26  # 类别数
    K = 5  # 混淆矩阵文件数

    try:
        # 读取并处理混淆矩阵
        M_cc_avg = load_confusion_matrices(directory, C, K)
        print("Average M_cc values:")
        print(M_cc_avg)

        # 计算 S_bal
        s_bal = compute_s_bal(M_cc_avg, C)
        print(f"S_bal(D) = {s_bal}")

    except Exception as e:
        print(f"Error: {e}")


def compute_s_lbl(A0, A_rand, p_list, A_pi_list, N_P):
    """
    计算 S_lbl(D; r*).

    参数:
    - A0: shuffle=0.0 时的 val_acc 最大值。
    - A_rand: 随机值 (1/26)。
    - p_list: p_i 列表 (N_P 个)。
    - A_pi_list: A(p_i) 列表 (N_P 个)。
    - N_P: p_i 的数量。

    返回:
    - S_lbl 值。
    """
    if N_P == 0:
        raise ValueError("No p_i values found")

    denominator = A0 - A_rand
    if denominator == 0:
        raise ValueError("A0 - A_rand = 0, cannot divide")

    sum_term = 0
    for p_i, A_pi in zip(p_list, A_pi_list):
        term = abs((A0 - A_pi) / denominator - p_i)
        sum_term += term

    s_lbl = 1 - (sum_term / N_P)
    return s_lbl


def load_sanity_check_files(directory, C=26):
    """
    从指定目录读取 sanity_check_shuffle*.csv 文件，提取 A0, p_i 和 A(p_i)。

    参数:
    - directory: 包含文件的目录路径。
    - C: 类别数，用于计算 A_rand = 1/C (默认26)。

    返回:
    - A0, A_rand, p_list, A_pi_list, N_P
    """
    # 获取所有匹配的文件
    files = glob.glob(os.path.join(directory, "sanity_check_shuffle*.csv"))

    if not files:
        raise FileNotFoundError(f"No files found in {directory} matching 'sanity_check_shuffle*.csv'")

    # 正则表达式提取 p_i 值
    p_pattern = re.compile(r"shuffle([0-9.]+)\.csv$")

    A0 = None
    p_list = []
    A_pi_list = []

    for file in files:
        # 提取 p_i
        p_match = p_pattern.search(file)
        if not p_match:
            print(f"Warning: Could not extract p_i from {file}")
            continue
        p_i = float(p_match.group(1))

        # 读取 CSV 的 val_acc 列并取最大值
        try:
            df = pd.read_csv(file)
            if 'val_acc' not in df.columns:
                print(f"Warning: 'val_acc' column not found in {file}, skipping")
                continue
            A = df['val_acc'].max()
            # A = df['val_acc'][80]
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        if p_i == 0.0:
            A0 = A
            print(f"Found A0 = {A0} from {file}")
        else:
            p_list.append(p_i)
            A_pi_list.append(A)

    N_P = len(p_list)
    if N_P != 5:
        raise ValueError(f"Expected 5 p_i values, found {N_P}")

    if A0 is None:
        raise ValueError("No shuffle=0.0 file found for A0")

    A_rand = 1 / C  # 随机准确率

    return A0, A_rand, p_list, A_pi_list, N_P


def main3():
    # 参数
    directory = "result\\sanity_check_1"  # 文件所在目录
    C = 14  # 类别数

    try:
        # 读取并处理文件
        A0, A_rand, p_list, A_pi_list, N_P = load_sanity_check_files(directory, C)
        print(f"A_rand = {A_rand}")
        print(f"p_list = {p_list}")
        print(f"A_pi_list = {A_pi_list}")

        # 计算 S_lbl
        s_lbl = compute_s_lbl(A0, A_rand, p_list, A_pi_list, N_P)
        print(f"S_lbl(D; r*) = {s_lbl}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
    main2()
    main3()