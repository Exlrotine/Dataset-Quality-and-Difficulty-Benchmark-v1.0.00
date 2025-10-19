import os
import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Callable

import mysql
import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader
# print(torch.version)
# print(torch.version.cuda)
# print(torch.cuda.is_available())

# 从MySQL表格里取数据标签及路径
def data_from_table(host="localhost", user="root", password="123456", database="pmsm_fault", table_name="motor1"):
    # root = '../dataset'
    all_data = []
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)  # 连接数据库
        cursor = connection.cursor()
        query = f"SELECT class, sub_class, magnitude, file_path, file_name  FROM {table_name}"  # 查询表中路径和标签
        cursor.execute(query)
        rows = cursor.fetchall()  # 获取所有结果
        for row in rows:
            class_name, sub_class, magnitude, file_path, file_name = row
            label_parts = [class_name]
            if sub_class:
                label_parts.append(sub_class)
            if magnitude is not None:  # magnitude可能为0，需检查是否为None
                label_parts.append(str(magnitude))
            label = '_'.join(label_parts)
            full_path = os.path.join(file_path, file_name)
            all_data.append((full_path, label))
        random.shuffle(all_data)
        return all_data

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None, None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()


def data_from_folder(data_folder: str) -> List[Tuple[str, str]]:
    all_data = []
    classes = []
    for root, dirs, files in os.walk(data_folder):
        relpath = os.path.relpath(root, data_folder)
        parts = relpath.split(os.sep)
        if len(parts) == 1 and parts[0] != '.':
            class_name = parts[0]
        elif len(parts) == 2:
            class_name = f"{parts[0]}_{parts[1]}"
        else:
            continue
        images = [os.path.join(root, f) for f in files if f.lower().endswith('.png')]
        if images:
            classes.append(class_name)
            all_data += [(p, class_name) for p in images]
    classes = sorted(set(classes))
    random.shuffle(all_data)
    return all_data


class GraphImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], img_to_graph: Callable, **kwargs):
        self.samples = samples
        self.img_to_graph = img_to_graph
        self.kw = kwargs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        g = self.img_to_graph(path, y, **self.kw)
        return g, torch.tensor(y, dtype=torch.long)


class All_Data_Graph(object):
    def __init__(self, data_root: str, train_ratio: float, img_to_graph: Callable, c_num: int = None, **graph_kwargs):
        # 1) 读取并分桶
        buckets: Dict[str, List[str]] = defaultdict(list)
        for path, cls in data_from_folder(data_root):
            buckets[cls].append(path)

        # 2) 选取类别子集（如需）
        classes = sorted(buckets.keys())
        if c_num is not None and c_num > 0:
            random.shuffle(classes)
            classes = classes[:min(c_num, len(classes))]

        # 3) 标签映射
        self.label_dict = {c: i for i, c in enumerate(sorted(classes))}
        self.inv_label_dict = {i: c for c, i in self.label_dict.items()}
        self.num_classes = len(self.label_dict)

        # 4) 样本切分
        self.train_samples: List[Tuple[str, int]] = []
        self.val_samples:   List[Tuple[str, int]] = []
        self.train_num_per_cls: Dict[int, int] = {}

        for c in classes:
            lbl_id = self.label_dict[c]
            paths = buckets[c][:]
            random.shuffle(paths)
            n_total = len(paths)
            n_train = max(1, math.ceil(n_total * train_ratio))
            self.train_num_per_cls[lbl_id] = n_train
            train_files, val_files = paths[:n_train], paths[n_train:]
            self.train_samples += [(p, lbl_id) for p in train_files]
            self.val_samples   += [(p, lbl_id) for p in val_files]

        # 5) 保存构图函数与参数
        self.img_to_graph = img_to_graph
        self.graph_kwargs = graph_kwargs

    def get_loader(self, split: str, batch_graphs: int = 32, shuffle: bool = True, num_workers: int = 0) -> GraphDataLoader:
        assert split in ('train', 'val')
        samples = self.train_samples if split == 'train' else self.val_samples
        dataset = GraphImageDataset(samples, self.img_to_graph, **self.graph_kwargs)
        # Windows 建议 num_workers=0；Linux 可酌情调大
        return GraphDataLoader(dataset, batch_size=batch_graphs, shuffle=shuffle, num_workers=num_workers)

