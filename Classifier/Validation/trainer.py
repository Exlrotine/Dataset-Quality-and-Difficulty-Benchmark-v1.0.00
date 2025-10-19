import argparse, json, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from ALL_Model.CNN_Classifier import CNNModel
from ALL_Model.ResNet_Classifier import ResNet18SE
from ALL_Model.MobileNet_Classifier import MobileNetV3Small
from ALL_Model.EfficientNet_Classifier import EfficientNetB0
from ALL_Model.ViT_Classifier import VisionTransformer
# from data_loader import get_data_loader, extract_classes
from dataloader_cifar import get_data_loader, extract_classes
from sklearn.metrics import f1_score



MODEL_REGISTRY = {"CNNModel": CNNModel, "ResNet18": ResNet18SE, "MobileNetV3": MobileNetV3Small,
                  "EfficientNet": EfficientNetB0, "ViT": VisionTransformer}


class Trainer:
    """训练管理类：读取 JSON 配置 → 构建模型 / 数据 → 训练 & 验证 → 记录结果"""

    def __init__(self, cfg: str, current_fold):
        script_dir = Path(__file__).parent
        config_dir = script_dir / ".config" # / "seed_ratio" # "ratio_model"
        cfg_path = config_dir / cfg
        with open(cfg_path, "r", encoding="utf-8") as f:  # 1. 读配置
            self.cfg = json.load(f)
        self.exp_name = self.cfg.get("name", Path(cfg_path).stem)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 2. 设备与随机种子
        self.seed = self.cfg.get("seed", 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.current_fold = current_fold
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self._build_dataloaders()  # 3. 数据
        self.model_name = self.cfg.get("model", "CNNModel")  # 4. 模型 / 优化器 / 调度器
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"模型 '{self.model_name}' 不在已注册模型列表中: {list(MODEL_REGISTRY.keys())}")
        keys = self.cfg.get("model_args", {})
        keys["num_classes"] = 10
        self.model = MODEL_REGISTRY[self.model_name](**keys).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"],
                                           weight_decay=self.cfg.get("weight_decay", 0.0))
        sched_cfg = self.cfg.get("scheduler", {})
        if sched_cfg:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=sched_cfg.get("step", 30),
                                                             gamma=sched_cfg.get("gamma", 0.1))
        else:
            self.scheduler = None
        self.log_dir = Path("result") / self.exp_name  # 5. 记录器
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # self.metrics_csv = self.log_dir / f"{self.exp_name}_seed{self.seed}_r{self.tr_ratio}.csv"
        # self.metrics_csv = self.log_dir / f"{self.exp_name}_shuffle{self.cfg["shuffle_ratio"]}.csv"
        # self.metrics_csv = self.log_dir / f"{self.exp_name}_m{self.cfg["model"]}_r{self.tr_ratio}.csv"
        # self.metrics_csv = self.log_dir / f"{self.exp_name}_m{self.cfg["model"]}_phi{keys["phi"]}_r{self.tr_ratio}.csv"
        self.metrics_csv = self.log_dir / f"{self.exp_name}_m{self.cfg["model"]}_a{keys["alpha"]}_b{keys["blocks"]}.csv"
        # self.metrics_csv = self.log_dir / f"{self.exp_name}_m{self.cfg["model"]}_r{self.tr_ratio}_k{self.current_fold}.csv"
        with open(self.metrics_csv, "w", encoding="utf-8") as f:
            # f.write("epoch,train_loss,train_eval_loss,val_loss,train_acc,train_eval_acc,val_acc,val_f1,lr,time\n")
            f.write("epoch,train_loss,val_loss,train_acc,val_acc,val_f1,lr,time\n")
        self.best_acc = 0.0
        self.best_path = self.log_dir / f"m{self.model_name}_r{self.tr_ratio}.pth"
        # self.best_path = self.log_dir / f"m{self.model_name}_all_k{self.current_fold}.pth"
        self.feature_maps = {}

    def _build_dataloaders(self):  # 数据加载
        self.tr_ratio = self.cfg.get("train_ratio", 0.9)
        self.if_rd = self.cfg.get("random_label", False)
        self.sf_r = self.cfg.get("shuffle_ratio", 0)
        self.bs = self.cfg["batch_size"]
        self.train_loader, self.val_loader = get_data_loader(train_ratio=self.tr_ratio, batch_size=self.bs,
                              random_label=self.if_rd, shuffle_ratio=self.sf_r, seed=self.seed)
        self.class_names = extract_classes(self.train_loader.dataset)

    def _run_epoch(self, loader: DataLoader, train: bool):  # 单个epoch
        if train:
            self.model.train()
        else:
            self.model.eval()
        running_loss, running_correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                if train:
                    loss.backward()
                    self.optimizer.step()
                preds = outputs.argmax(dim=1)   # softmax一样
                running_loss += loss.item() * imgs.size(0)
                running_correct += (preds == labels).sum().item()
                n += imgs.size(0)
        epoch_loss = running_loss / n
        epoch_acc = running_correct / n
        return epoch_loss, epoch_acc

    def fit(self):  # 主训练循环
        epochs = self.cfg["epochs"]
        val_freq = self.cfg.get("val_freq", 1)
        patience = self.cfg.get("patience", 5)
        dt = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()  # 训练epoch
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            # te_loss, te_acc = self._run_epoch(self.train_loader, train=False)
            if self.scheduler:
                self.scheduler.step()
            if epoch % val_freq == 0 or epoch == epochs:  # 测试epoch
                val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            else:
                val_loss = val_acc = float("nan")
            y_true, y_pred = [], []  # 计算精确率和召回率
            with torch.no_grad():
                for x, y in self.val_loader:
                    logits = self.model(x.to(self.device))
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(logits.argmax(1).cpu().numpy())
            val_f1 = f1_score(y_true, y_pred, average='macro')  # 计算F1分数
            current_lr = self.optimizer.param_groups[0]["lr"]  # 学习率
            with open(self.metrics_csv, "a", encoding="utf-8") as f:  # 写入CSV表格
                f.write(
                    # f"{epoch},{tr_loss:.6f},{te_loss:.4f},{val_loss:.6f},{tr_acc:.4f},{te_acc:.4f},"
                    f"{epoch},{tr_loss:.6f},{val_loss:.6f},{tr_acc:.4f},"
                    f"{val_acc:.4f},{val_f1:.4f},{current_lr:.8f},{dt:.3f}\n")
            if not np.isnan(val_acc) and val_acc > self.best_acc:  # 准确率最佳时执行
                self.best_acc = val_acc
                # torch.save(self.model.state_dict(), self.best_path)
                epochs_no_improve = 0      # 早停
                cm = confusion_matrix(y_true, y_pred, normalize='true')  # 混淆矩阵（行 = 1）
                df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
                # out = Path(self.log_dir / f"cm_{self.cfg['name']}_seed{self.seed}_k{self.current_fold}_r{self.tr_ratio}.csv")
                out = Path(self.log_dir / f"cm_{self.cfg['name']}_seed{self.seed}_r{self.tr_ratio}.csv")  # 3) 保存为 csv
                # out = Path(self.log_dir / f"cm_{self.cfg['name']}_shuffle{self.cfg["shuffle_ratio"]}.csv")  # 3) 保存为 csv
                # out = Path(self.log_dir / f"cm_{self.cfg['name']}_m{self.cfg["model"]}_r{self.tr_ratio}.csv")  # 3) 保存为 csv
                # out = Path(self.log_dir / f"cm_{self.cfg['name']}_m{self.cfg["model"]}_k{self.current_fold}.csv")  # 3) 保存为 csv
                df.to_csv(out, float_format='%.6f')
            else:
                epochs_no_improve += 1      # 早停
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch} epochs (patience: {patience}).")
                    break

            dt = time.time() - t0
            print(f"[Epoch {epoch}/{epochs}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_acc={tr_acc * 100:.2f}% val_acc={val_acc * 100:.2f}%")  # f" time={dt:.1f}s")
        # print(f"Training completed. best_val_acc={self.best_acc * 100:.2f}%  saved=>{self.best_path}")
        print(f"Training completed. best_val_acc={self.best_acc * 100:.2f}% ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=["text.json"], type=str, nargs='+', help="Path to config JSON")
    args = parser.parse_args()

    for config in args.cfg:
        print(f"Starting training with config: {config}")
        trainer = Trainer(config,1)
        trainer.fit()
        print(f"Finished training with config: {config}\n")
