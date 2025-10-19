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


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, use_swish):
        super(MBConvBlock, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = []
        # Expansion
        if exp_channels != in_channels:
            layers.append(nn.Conv2d(in_channels, exp_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(exp_channels))
            layers.append(nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True))
        # Depthwise
        layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, kernel_size // 2, groups=exp_channels, bias=False))
        layers.append(nn.BatchNorm2d(exp_channels))
        layers.append(nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True))
        # SE block
        if use_se:
            layers.append(SEBlock(exp_channels))
        # Projection
        layers.append(nn.Conv2d(exp_channels, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
        self.drop_connect_rate = 0.2 if self.use_res_connect else 0.0

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            if self.drop_connect_rate > 0 and self.training:
                out = out * torch.rand(out.size(0), 1, 1, 1, device=out.device).ge(self.drop_connect_rate).float() / (1 - self.drop_connect_rate)
            return x + out
        return out


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=13, phi=0):
        super(EfficientNetB0, self).__init__()
        # EfficientNet-B0 configuration: (in_channels, exp_channels, out_channels, kernel_size, stride, use_se, use_swish, repeats)
        alpha, beta = 1.2, 1.1
        width = alpha ** phi
        depth = beta ** phi
        config = [
            (int(16* width), int(16 * width), int(16* width), 3, 1, True, False, 1), # 1223341
            (int(16* width), int(96 * width), int(24* width), 3, 2, True, False, int(2 * depth)),
            (int(24* width), int(144 * width), int(40* width), 5, 2, True, False, int(2 * depth)),
            (int(40* width), int(240 * width), int(80* width), 3, 2, True, True, int(3 * depth)),
            (int(80* width), int(480 * width), int(112* width), 5, 1, True, True, int(3 * depth)),
            (int(112* width), int(672 * width), int(192* width), 5, 2, True, True, int(4 * depth)),
            (int(192* width), int(1152 * width), int(320* width), 3, 1, True, True, int(1 * depth)),
        ]
        # Stem
        self.conv1 = nn.Conv2d(3, int(32 * width), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * width))
        self.swish1 = nn.SiLU(inplace=True)
        # Blocks
        self.blocks = nn.Sequential()
        in_channels = int(32 * width)
        for idx, (in_c, exp_c, out_c, k, s, se, swish, r) in enumerate(config):
            for i in range(r):
                stride = s if i == 0 else 1
                self.blocks.add_module(f"block{idx}_{i}", MBConvBlock(in_channels, exp_c, out_c, k, stride, se, swish))
                in_channels = out_c
        # Head
        self.conv2 = nn.Conv2d(in_channels, int(1280 * width), kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(1280 * width))
        self.swish2 = nn.SiLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(1280 * width), int(128 * width)),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(int(128 * width), num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.swish1(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.swish2(self.bn2(self.conv2(x)))
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
