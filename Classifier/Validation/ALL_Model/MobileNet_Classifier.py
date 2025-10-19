import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = []
        # Expansion
        if exp_channels != in_channels:
            layers.append(nn.Conv2d(in_channels, exp_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(exp_channels))
            layers.append(nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True))
        # Depthwise
        layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, kernel_size // 2, groups=exp_channels, bias=False))
        layers.append(nn.BatchNorm2d(exp_channels))
        layers.append(nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True))
        # SE block
        if use_se:
            layers.append(SEBlock(exp_channels))
        # Projection
        layers.append(nn.Conv2d(exp_channels, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=26):
        super(MobileNetV3Small, self).__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.Hardswish(inplace=True)
        # Inverted residual blocks (MobileNetV3-Small configuration)
        self.blocks = nn.Sequential(
            # in, exp, out, kernel, stride, se, hs
            InvertedResidual(16, 16, 16, 3, 2, True, False),  # SE, ReLU
            InvertedResidual(16, 72, 24, 3, 2, False, False),  # ReLU
            InvertedResidual(24, 88, 24, 3, 1, False, False),  # ReLU
            InvertedResidual(24, 96, 40, 5, 2, True, True),   # SE, Hardswish
            InvertedResidual(40, 240, 40, 5, 1, True, True),   # SE, Hardswish
            InvertedResidual(40, 240, 40, 5, 1, True, True),   # SE, Hardswish
            InvertedResidual(40, 120, 48, 5, 1, True, True),   # SE, Hardswish
            InvertedResidual(48, 144, 48, 5, 1, True, True),   # SE, Hardswish
            InvertedResidual(48, 288, 96, 5, 2, True, True),   # SE, Hardswish
            InvertedResidual(96, 576, 96, 5, 1, True, True),   # SE, Hardswish
            InvertedResidual(96, 576, 96, 5, 1, True, True),   # SE, Hardswish
        )
        # Head
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = nn.Hardswish(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.hs2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
