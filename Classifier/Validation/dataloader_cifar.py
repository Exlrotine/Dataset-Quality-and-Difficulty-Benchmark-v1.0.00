import os
import random
import pickle
import numpy as np

import mysql.connector  # 若无用可删
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import imagehash  # 若无用可删
import torch


# ====================== CIFAR-10 读取 ======================

def _load_cifar_batch(file_path):
    """
    读取单个 CIFAR 批次，返回:
      images: (N, 32, 32, 3) uint8
      labels: List[int]
      filenames: List[str]
    """
    with open(file_path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')

    data = d[b'data']  # (N, 3072) = 1024R + 1024G + 1024B
    labels = d.get(b'labels', d.get(b'fine_labels', []))  # 兼容 CIFAR-100 的字段名
    filenames = [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn)
                 for fn in d.get(b'filenames', [])]

    # 变形为 (N, 3, 32, 32) -> (N, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels, filenames


def _load_label_names(data_dir):
    """
    尝试从 batches.meta 读取类别名；若不存在则回退为标准 CIFAR-10 类别。
    """
    meta_path = os.path.join(data_dir, 'batches.meta')
    default = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f, encoding='bytes')
            names = [n.decode('utf-8') for n in meta.get(b'label_names', [])]
            if names:
                return names
        except Exception:
            pass
    return default


def load_cifar10(data_dir='..\\cifar-10\\'):
    """
    读取 CIFAR-10:
      - 训练 = data_batch_1~5 (共 50k)
      - 测试 = test_batch (10k)
    返回: (train_imgs, train_labels, train_fns), (test_imgs, test_labels, test_fns), classes
    """
    train_imgs_list, train_labels_list, train_fns_list = [], [], []
    for i in range(1, 6):
        path = os.path.join(data_dir, f'data_batch_{i}')
        imgs, labels, fns = _load_cifar_batch(path)
        train_imgs_list.append(imgs)
        train_labels_list.extend(labels)
        train_fns_list.extend(fns)

    train_imgs = np.concatenate(train_imgs_list, axis=0)

    test_path = os.path.join(data_dir, 'test_batch')
    test_imgs, test_labels, test_fns = _load_cifar_batch(test_path)

    classes = _load_label_names(data_dir)
    return (train_imgs, np.array(train_labels_list), train_fns_list), \
           (test_imgs, np.array(test_labels), test_fns), \
           classes


# ====================== 数据集封装 ======================

class CIFARDataset(Dataset):
    """
    与原来的 ImageDataset 对齐：__getitem__ 返回 (image_tensor, label_int)
    并提供 .classes / .class_to_idx
    """
    def __init__(self, images, labels, filenames=None, classes=None, transform=None):
        # images: (N, 32, 32, 3) uint8
        self.images = images
        self.labels = labels.astype(int).tolist() if isinstance(labels, np.ndarray) else list(labels)
        self.filenames = filenames if filenames is not None else [f'{i}.png' for i in range(len(self.labels))]
        self.transform = transform if transform else transforms.ToTensor()
        self.classes = list(classes) if classes is not None else sorted(set(map(str, self.labels)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (32, 32, 3), uint8
        pil = Image.fromarray(img)
        image = self.transform(pil)  # 默认 ToTensor -> (3, 32, 32) float32 in [0,1]
        label = int(self.labels[idx])  # 0~9
        return image, label


# ========== 兼容：保留随机标签封装 & extract_classes 不变 ==========

class RandomLabelDataset(Dataset):
    def __init__(self, base_ds, shuffle_ratio=1.0, seed=42):
        self.base_ds = base_ds
        self.shuffle_idx = list(range(len(base_ds)))
        random.Random(seed).shuffle(self.shuffle_idx)
        cutoff = int(len(self.shuffle_idx) * shuffle_ratio)
        self.shuffle_set = set(self.shuffle_idx[:cutoff])

        # 准备一个随机 label 列表（与 base_ds 长度对齐）
        labels = [lbl for _, lbl in base_ds]
        random.Random(seed + 1).shuffle(labels)
        self.rand_labels = labels

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        if idx in self.shuffle_set:
            label = self.rand_labels[idx]
        return img, label


def extract_classes(dataset):
    while True:
        if hasattr(dataset, "classes"):
            return dataset.classes
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
            continue
        if hasattr(dataset, "base_ds"):
            dataset = dataset.base_ds
            continue
        raise AttributeError("Cannot find .classes in dataset chain")


# ====================== 对外函数（保持签名，行为按你要求改） ======================

def build_full_dataset(random_label=False, shuffle_ratio=1.0, seed=42, data_dir='..\\cifar-10\\'):
    """
    返回“训练部分”的完整 Dataset（供 K-Fold 等使用），同时给出一个 data 列表
    来兼容你原来的返回结构。这里 data = [(filename, class_name), ...]。
    """
    (tr_imgs, tr_labels, tr_fns), _, classes = load_cifar10(data_dir)
    dataset = CIFARDataset(tr_imgs, tr_labels, tr_fns, classes, transform=transforms.ToTensor())
    if random_label:
        dataset = RandomLabelDataset(dataset, shuffle_ratio, seed)

    # 兼容你原来 data_from_folder() 的第二返回值形态
    idx2name = {i: name for i, name in enumerate(classes)}
    data = list(zip(tr_fns, [idx2name[int(y)] for y in tr_labels]))
    return dataset, data


def get_data_loader(train_ratio=0.9, batch_size=16, random_label=False,
                    shuffle_ratio=1.0, seed=42, data_dir='..\\cifar-10\\'):
    """
    返回 (train_loader, val_loader)
    - 训练集：CIFAR-10 的 data_batch_1~5，先构建完整训练集，再按 train_ratio 抽样
    - 验证/测试集：固定使用 test_batch
    - random_label: 若为 True，仅对“抽样后的训练子集”按 shuffle_ratio 打乱标签
    """
    # 读入完整 CIFAR-10
    (tr_imgs, tr_labels, tr_fns), (te_imgs, te_labels, te_fns), classes = load_cifar10(data_dir)

    # 完整训练集
    full_train_ds = CIFARDataset(tr_imgs, tr_labels, tr_fns, classes, transform=transforms.ToTensor())

    # —— 在训练集上按比例随机抽样 —— #
    if not (0 < train_ratio <= 1.0):
        raise ValueError("train_ratio must be in (0, 1].")
    train_ds = full_train_ds
    if train_ratio < 1.0:
        n = len(full_train_ds)
        m = max(1, int(round(n * train_ratio)))
        g = torch.Generator()
        g.manual_seed(seed)                  # 可复现
        idx = torch.randperm(n, generator=g)[:m].tolist()
        from torch.utils.data import Subset
        train_ds = Subset(full_train_ds, idx)  # Subset 保留了对底层 .classes 的可追溯性，extract_classes 仍可用

    # 仅对“抽样后的训练集”打乱标签（若启用）
    if random_label:
        train_ds = RandomLabelDataset(train_ds, shuffle_ratio, seed)

    # 固定测试集
    val_ds = CIFARDataset(te_imgs, te_labels, te_fns, classes, transform=transforms.ToTensor())

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    return train_loader, val_loader

