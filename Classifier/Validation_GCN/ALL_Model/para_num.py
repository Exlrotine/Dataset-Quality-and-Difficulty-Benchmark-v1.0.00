from CNN_Classifier import CNNModel
from EfficientNet_Classifier import EfficientNetB0
from MobileNet_Classifier import MobileNetV3Small
from ResNet_Classifier import ResNet18SE
from ViT_Classifier import VisionTransformer
import torch.nn as nn

CNNmodel = CNNModel(num_classes=13)
print(sum(p.numel() for p in CNNmodel.parameters()))

MobileNet = MobileNetV3Small(num_classes=13)
print(sum(p.numel() for p in MobileNet.parameters()))

ResNet = ResNet18SE(num_classes=13, alpha=16,blocks=2)
print(sum(p.numel() for p in ResNet.parameters()))
for name, p in ResNet.named_parameters():
    if p.numel() == 0:
        print(name, p.shape)

EfficienNet = EfficientNetB0(num_classes=13, phi=3)
print(sum(p.numel() for p in EfficienNet.parameters()))
print(sum(1 for module in EfficienNet.modules() if isinstance(module, nn.Conv2d)))
print(sum(1 for _ in EfficienNet.modules()))
for name, p in EfficienNet.named_parameters():
    if p.numel() == 0:
        print(name, p.shape)

ViTmodel = VisionTransformer(num_classes=13)
print(sum(p.numel() for p in ViTmodel.parameters()))
