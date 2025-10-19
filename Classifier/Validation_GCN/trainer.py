
# -*- coding: utf-8 -*-
import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from functools import partial
import file_to_graph as f2g
from data_loader import All_Data_Graph
from ALL_Model.GCN_Classifier import GraphGATClassifier

MODEL_REGISTRY = {"GNNModule": GraphGATClassifier}

class Trainer:
    def __init__(self, cfg: str, current_fold: int = 1):
        script_dir = Path(__file__).parent
        config_dir = script_dir / ".config"
        cfg_path = config_dir / cfg
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.exp_name = self.cfg.get("name", Path(cfg_path).stem)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = int(self.cfg.get("seed", 42))
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)

        self.current_fold = current_fold
        self._build_dataloaders()

        self.model_name = self.cfg.get("model", "GraphGATClassifier")
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"模型 '{self.model_name}' 未注册，可选: {list(MODEL_REGISTRY.keys())}")
        margs = dict(self.cfg.get("model_args", {}))
        margs.setdefault("in_feats", 20)  # file_to_graph_v2 的节点特征维度
        margs.setdefault("num_classes", self.num_classes)
        self.model = MODEL_REGISTRY[self.model_name](**margs).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.cfg.get("lr", 1e-3)),
                                           weight_decay=float(self.cfg.get("weight_decay", 0.0)))
        sched_cfg = self.cfg.get("scheduler", {})
        self.scheduler = (torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=int(sched_cfg.get("step", 30)),
                                                          gamma=float(sched_cfg.get("gamma", 0.1)))
                          if sched_cfg else None)

        self.log_dir = Path("result") / self.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = self.log_dir / f"{self.exp_name}_m{self.model_name}_r{self.tr_ratio}_k{self.current_fold}.csv"
        with open(self.metrics_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc,val_f1,lr,time\n")

        self.best_acc = 0.0
        self.best_path = self.log_dir / f"m{self.model_name}_all_k{self.current_fold}.pth"

    def _build_dataloaders(self):
        self.tr_ratio = float(self.cfg.get("train_ratio", 0.8))
        self.bs = int(self.cfg.get("batch_size", 32))

        all_data = All_Data_Graph(
            data_root=self.cfg.get("data_root", "..\\dataset\\"),
            train_ratio=self.tr_ratio,
            img_to_graph=partial(f2g.img_to_graphs_cached, cache_dir=r"D:\tf_graph_cache"),
            n_segments=int(self.cfg.get("n_segments", 32)),
            add_knn=bool(self.cfg.get("add_knn", True)),
            knn_k=int(self.cfg.get("knn_k", 4)),
            device='cpu'
        )
        self.train_loader = all_data.get_loader('train', batch_graphs=self.bs, shuffle=True,  num_workers=0)
        self.val_loader   = all_data.get_loader('val',   batch_graphs=self.bs, shuffle=False, num_workers=0)
        self.num_classes  = all_data.num_classes

    @torch.no_grad()
    def _check_shapes(self, logits, labels, where="train"):
        if logits.ndim == 4:
            # 典型错误：N,C,H,W；CrossEntropy 会要求 labels 形状是 N,H,W
            # 但我们要的是图级分类：必须是 2D 的 [N, C]
            raise RuntimeError(f"[{where}] 你的模型输出 logits 形状是 {tuple(logits.shape)}，"
                               f"看起来像是空间输出(N,C,H,W)。请改为图级输出 [N, C]。")
        if logits.ndim != 2:
            raise RuntimeError(f"[{where}] 期望 logits 是 2D [N, C]，但拿到 {logits.ndim}D 形状 {tuple(logits.shape)}。")
        if labels.ndim != 1:
            raise RuntimeError(f"[{where}] 期望 labels 是 1D [N]，但拿到 {labels.ndim}D 形状 {tuple(labels.shape)}。")

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        running_loss, running_correct, n = 0.0, 0, 0
        all_true, all_pred = [], []
        for bg, labels in loader:
            bg = bg.to(self.device); labels = labels.to(self.device)
            with torch.set_grad_enabled(train):
                logits = self.model(bg)  # [B, C]
                self._check_shapes(logits, labels, where="train" if train else "val")
                loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            preds = logits.argmax(dim=1)
            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_correct += (preds == labels).sum().item()
            n += bs
            if not train:
                all_true.append(labels.detach().cpu())
                all_pred.append(preds.detach().cpu())
        epoch_loss = running_loss / max(1, n)
        epoch_acc = running_correct / max(1, n)
        if not train and len(all_true) > 0:
            import torch as _t
            y_true = _t.cat(all_true).numpy()
            y_pred = _t.cat(all_pred).numpy()
        else:
            y_true = y_pred = None
        return epoch_loss, epoch_acc, y_true, y_pred

    def fit(self):
        epochs = int(self.cfg.get("epochs", 50))
        val_freq = int(self.cfg.get("val_freq", 1))
        patience = int(self.cfg.get("patience", 10))
        dt = 0.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, _1, _2 = self._run_epoch(self.train_loader, train=True)
            if self.scheduler: self.scheduler.step()

            if epoch % val_freq == 0 or epoch == epochs:
                val_loss, val_acc, y_true, y_pred = self._run_epoch(self.val_loader, train=False)
                val_f1 = f1_score(y_true, y_pred, average='macro') if y_true is not None else float('nan')
            else:
                val_loss = val_acc = val_f1 = float('nan')

            current_lr = self.optimizer.param_groups[0]["lr"]
            with open(self.metrics_csv, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{tr_loss:.6f},{val_loss:.6f},{tr_acc:.4f},{val_acc:.4f},{val_f1:.4f},{current_lr:.8f},{dt:.3f}\n")

            if not np.isnan(val_acc) and val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.best_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch} epochs (patience: {patience}).")
                    break

            dt = time.time() - t0
            print(f"[Epoch {epoch}/{epochs}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_acc={tr_acc*100:.2f}% val_acc={val_acc*100:.2f}%")

        print(f"Training completed. best_val_acc={self.best_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=["k_fold.JSON"], type=str, nargs='+', help="Path to config JSON (under .config)")
    args = parser.parse_args()

    for config in args.cfg:
        print(f"Starting training with config: {config}")
        trainer = Trainer(config, 1)
        trainer.fit()
        print(f"Finished training with config: {config}\n")
