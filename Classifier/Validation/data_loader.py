import os
import random
import mysql.connector
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import imagehash
import torch
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


def data_from_folder(data_folder='..\\dataset\\'):
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
            continue  # 跳过更深的层级或根目录
        images = [os.path.join(root, img) for img in files if img.endswith('.png')]
        if images:
            classes.append(class_name)
            all_data += [(img, class_name) for img in images]
    classes = sorted(set(classes))
    random.shuffle(all_data)
    return all_data


class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # [(img_path, class_name)]
        self.transform = transform if transform else transforms.ToTensor()
        self.classes = sorted(list(set([item[1] for item in data])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]  # Convert to index for model
        image = self.transform(image)
        return image, label


# === 新增：一次性拿到完整数据集，用于 k-fold ===
def build_full_dataset(random_label=False, shuffle_ratio=1.0, seed=42):
    """   返回完整 ImageDataset（不拆 train/val）， 供外部 StratifiedKFold 按索引切分  """
    data = data_from_folder()                 # or data_from_folder()
    dataset = ImageDataset(data, transform=transforms.ToTensor())
    if random_label:                         # 是否做标签打乱实验
        dataset = RandomLabelDataset(dataset, shuffle_ratio, seed)
    return dataset, data


def get_data_loader(train_ratio=0.9, batch_size=16, random_label=False,  shuffle_ratio=1.0, seed=42):
    """一次性返回训练和验证数据加载器，确保数据不重叠"""
    all_data = data_from_folder()
    all_dataset = ImageDataset(all_data, transform=transforms.Compose([# transforms.Resize(200*200),
                               transforms.ToTensor()]))
    torch.manual_seed(seed)       # 设置随机种子确保可重现
    train_size = int(len(all_dataset) * train_ratio)
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])  # 一次性划分，确保无重叠

    if random_label:
        train_dataset = RandomLabelDataset(train_dataset, shuffle_ratio, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    return train_loader, val_loader


class RandomLabelDataset(Dataset):
    def __init__(self, base_ds, shuffle_ratio=1.0, seed=42):
        self.base_ds = base_ds
        self.shuffle_idx = list(range(len(base_ds)))
        random.Random(seed).shuffle(self.shuffle_idx)
        # 只对前 shuffle_ratio 部分打乱
        cutoff = int(len(self.shuffle_idx) * shuffle_ratio)
        self.shuffle_set = set(self.shuffle_idx[:cutoff])

        # 准备一个随机 label 列表：与 base_ds.classes 对齐
        labels = [lbl for _, lbl in base_ds]          # 原标签 idx
        random.Random(seed+1).shuffle(labels)         # 打乱
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
        if hasattr(dataset, "dataset"):       # Subset / ConcatDataset
            dataset = dataset.dataset
            continue
        if hasattr(dataset, "base_ds"):       # RandomLabelDataset
            dataset = dataset.base_ds
            continue
        raise AttributeError("Cannot find .classes in dataset chain")