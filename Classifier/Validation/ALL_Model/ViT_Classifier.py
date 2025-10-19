import torch
import torch.nn as nn
import torch.nn.functional as F

# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=200, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.se = SEBlock(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        # Project to patches
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        # x = self.se(x)  # Apply SE block
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        b = x.size(0)
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        return x


class TransformerEncoder(nn.Module):  #
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=200, patch_size=16, num_classes=26, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.encoder = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(128, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.patch_embed(x)  # (B, num_patches + 1, embed_dim)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        x = x[:, 0]  # Take cls token
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 尝试修改线性层的初始化
                nn.init.xavier_uniform_(m.weight)  # 尝试 Xavier Uniform
                # 或者 nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=0.02)  # 保持不变，或微调 std


if __name__ == "__main__":
    model = VisionTransformer(num_classes=26)
    print(model)