import argparse, json
from pathlib import Path
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ALL_Model.CNN_Classifier     import CNNModel
from ALL_Model.ResNet_Classifier  import ResNet18SE
from ALL_Model.MobileNet_Classifier import MobileNetV3Small
from ALL_Model.EfficientNet_Classifier import EfficientNetB0
from ALL_Model.ViT_Classifier     import VisionTransformer
import torchvision.utils as vutils
import cv2, torch


MODEL_REGISTRY = {
    "CNNModel": CNNModel, "ResNet18": ResNet18SE, "MobileNetV3": MobileNetV3Small,
    "EfficientNet": EfficientNetB0, "ViT": VisionTransformer
}

# 指定每个骨架模型用于 Grad-CAM 的“最后卷积层”
TARGET_LAYER = {
    "CNNModel":        lambda m: m.conv3,                   # 3×CNN
    "ResNet18":        lambda m: m.layer2[1].conv2,        # BasicBlock 2
    "MobileNetV3":     lambda m: m.blocks[2],                  # 倒残差出口
    "EfficientNet":    lambda m: m.blocks[2],               # 最后 MBConv
    # "ViT":             lambda m: m.norm,                  # ViT 用 CVT 模式
}

# ----- 预处理保持与训练一致 -------------------
preproc = T.Compose([T.Resize((200, 200)), T.ToTensor()])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        # 注册钩子
        target_layer.register_forward_hook(self.save_features)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        target = model_output[0][target_class]
        target.backward()
        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.sum(weights[:, :, None, None] * self.features, dim=1)
        cam = F.relu(cam)  # ReLU激活
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam.detach().cpu().numpy()


def save_feature_grid(fmap, out_path):
    fmap = fmap[:16]  # 最多画前16张图
    fmap = fmap[:, :16]  # 每张图的前16通道
    fmap = fmap.reshape(-1, 1, fmap.shape[2], fmap.shape[3]).cpu()  # (16x16, 1, H, W)
    grid = vutils.make_grid(fmap, nrow=16, normalize=True, scale_each=True)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title("Feature Map Grid (first 16 images × 16 channels)")
    plt.imshow(grid.permute(1, 2, 0))       # C,H,W → H,W,C
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_heatmap_grid(cam_list, out_path, nrow=4):
    """   将灰度 CAM 列表着色后拼网格保存   """
    canvases = []
    for cam in cam_list[:16]:                         # 最多 16 张
        heatmap = cv2.applyColorMap(
            np.uint8(cam * 255), cv2.COLORMAP_JET)    # (H, W, 3) BGR 0-255
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(heatmap).permute(2, 0, 1).float() / 255.0
        canvases.append(tensor)

    grid = vutils.make_grid(torch.stack(canvases), nrow=nrow, normalize=False)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Grad-CAM Heatmap Grid")
    plt.imshow(grid.permute(1, 2, 0))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@torch.no_grad()
def load_images(paths):
    imgs_rgb, tensors = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img_rgb = T.ToTensor()(T.Resize((200, 200))(img)).permute(1,2,0).numpy()
        imgs_rgb.append(img_rgb)                # un-normalised 0-1 for heatmap
        tensors.append(preproc(img))
    return torch.stack(tensors), imgs_rgb


def build_model(cfg_path, device):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    name = cfg.get("model", "CNNModel")
    model = MODEL_REGISTRY[name](**cfg.get("model_args", {}))
    exp_name = cfg.get("name", None)
    ckpt_path = Path(__file__).parent / "result" / exp_name / f"m{name}_r{cfg.get("train_ratio", "0.8")}.pth"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, name


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mdl_name = build_model(args.cfg, device)
    imgs, imgs_rgb = load_images(args.images)
    imgs = imgs.to(device)

    # ----- Grad-CAM ---------------------------
    target_layer = TARGET_LAYER[mdl_name](model)
    cam = GradCAM(model, target_layer)
    grayscale_cam = []
    preds = model(imgs)
    for i in range(imgs.shape[0]):
        target_cls = args.cls if args.cls >= 0 else preds[i].argmax().item()
        cam_img = cam.generate_cam(imgs[i].unsqueeze(0), target_cls)  # 单张图
        grayscale_cam.append(cam_img[0])  # 注意 generate_cam 返回 shape: (1, H, W)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    # ----- Feature-Map 提取示例 -----------------
    with torch.no_grad():
        _ = model(imgs)              # forward once
    fmap = cam.features              # pytorch-grad-cam hook 用 .output

    # # 保存特征 Tensor 到 .pt 文件
    # feature_tensor_path = out_dir / f"{mdl_name}_features.pt"
    # torch.save(fmap.cpu(), feature_tensor_path) # 将特征移到 CPU 并保存
    # print(f"[INFO] Feature tensor saved to {feature_tensor_path.resolve()}")

    save_feature_grid(fmap, out_dir / f"{mdl_name}_fmap_grid.png")
    save_heatmap_grid(grayscale_cam, out_dir / f"{mdl_name}_gradcam_grid.png")
    print(f"[INFO] saved Grad-CAM & feature maps to {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",  required=True, help="path to training config JSON")
    parser.add_argument("--images", nargs='+', required=True, help="list of images for CAM")
    parser.add_argument("--cls", type=int, default=-1, help="目标类别索引；-1 表示用预测最大类")
    parser.add_argument("--out", default= "result\\imgs", help="output folder")
    args = parser.parse_args()
    main(args)

