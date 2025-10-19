import argparse, json
from pathlib import Path
import torch, torch.nn.functional as F
from ALL_Model.CNN_Classifier     import CNNModel
from ALL_Model.ResNet_Classifier  import ResNet18SE
from ALL_Model.MobileNet_Classifier import MobileNetV3Small
from ALL_Model.EfficientNet_Classifier import EfficientNetB0
from ALL_Model.ViT_Classifier     import VisionTransformer
from data_loader import get_data_loader                # ← 你的函数所在文件


MODEL_REGISTRY = {
    "CNNModel": CNNModel, "ResNet18": ResNet18SE, "MobileNetV3": MobileNetV3Small,
    "EfficientNet": EfficientNetB0, "ViT": VisionTransformer
}

# 指定每个骨架模型用于 Grad-CAM 的“最后卷积层”
TARGET_LAYER = {
    "CNNModel":        lambda m: m.conv3,                    # 3×CNN
    # "ResNet18":        lambda m: m.layer2[1].conv2,        # 画grad_CAM的那层
    "ResNet18":        lambda m: m.layer4[-1],              # features
    "MobileNetV3":     lambda m: m.blocks[-1],              # 倒残差出口
    "EfficientNet":    lambda m: m.blocks[-1],              # 最后 MBConv
    # "ViT":             lambda m: m.norm,                  # ViT 用 CVT 模式
}


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. 构建模型并加载权重 ----------
    cfg  = json.load(open(args.cfg, encoding="utf-8"))
    name = cfg.get("model", "CNNModel")
    model = MODEL_REGISTRY[name](**cfg.get("model_args", {}))
    exp_name = cfg.get("name", None)
    ckpt = Path(__file__).parent / "result" / exp_name / f"m{name}_r{cfg.get("train_ratio", "0.8")}.pth"
    # ckpt = Path(__file__).parent / "result" / "capacity"/ f"m{name}_alpha4_b1_epoch{cfg.get("epochs")}.pth"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # ---------- 2. 注册 forward hook 抓特征 ----------
    feats_list, labels_list = [], []

    def hook(_, __, output):     # output: (B, C, H, W)
        feats_list.append(output.detach().cpu())
    handle = TARGET_LAYER[name](model).register_forward_hook(hook)

    # ---------- 3. 遍历数据 ----------
    loader, _ = get_data_loader(train_ratio=1, batch_size=args.batch_size, random_label=False, shuffle_ratio=1.0)
    for bi, (x, y) in enumerate(loader):
        _ = model(x.to(device))
        labels_list.append(y)

    handle.remove()   # 摘钩

    # ---------- 4. 整理并保存 ----------
    feats = torch.cat(feats_list, 0)                 # (N, C, H, W)
    labels = torch.cat(labels_list, 0)               # (N,)
    out_dir = Path(args.out)                              # 先保留原始 4-D，方便可视化 / Grad-CAM
    out_dir.mkdir(exist_ok=True, parents=True)

    raw_path =  out_dir / f"m{name}_batch{args.batch_size}.raw.pt"
    torch.save({"feat": feats, "label": labels}, raw_path)

    # 再做 GAP 得到 (N, C) ——> 用于 UMAP / t-SNE
    feats_gap = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)  # (N, C)
    flat_path =  out_dir / f"m{name}_batch{args.batch_size}.gap.pt"
    torch.save({"feat": feats_gap, "label": labels}, flat_path)

    print(f"[√] Raw feature tensor  : {feats.shape}  →  {raw_path}")
    print(f"[√] GAP-pooled tensor   : {feats_gap.shape}  →  {flat_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",   required=True, help="训练时的 config JSON")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default= "result\\imgs", help="output folder")
    args = parser.parse_args()
    main(args)
