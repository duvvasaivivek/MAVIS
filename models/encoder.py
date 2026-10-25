"""
CNN Encoder (PyTorch)
=====================
Continual-ResNet-18 Feature Extractor.
Built completely from scratch to ensure:
1. No pretrained weights are used.
2. Optimized for 64x64 Tiny ImageNet inputs (no aggressive early pooling).
3. Uses GroupNorm instead of BatchNorm to prevent catastrophic forgetting via stat contamination.
"""

import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention block.
    Dynamically recalibrates channel-wise feature responses by modeling
    interdependencies between convolutional channels (Hu et al., CVPR 2018).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, planes)
        self.se = SEBlock(planes, reduction=16)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, self.expansion * planes)
            )

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CNNEncoder(nn.Module):
    """
    ResNet-18 variant designed from scratch for Continual Learning on 64x64 images.
    """
    def __init__(self, feature_dim: int = 512, dropout_rate: float = 0.0, base_width: int = 128):
        super(CNNEncoder, self).__init__()
        # base_width controls the channel count of the stem and stage 1; stages 2-4 use 2x, 4x, 8x.
        # The published runs used base_width=128 (channels 128/256/512/1024). GroupNorm(32) requires
        # every channel count to be a multiple of 32.
        assert base_width % 32 == 0 and base_width > 0, "base_width must be a positive multiple of 32"
        self.base_width = base_width
        self.in_planes = base_width

        # Initial layer: 3x3 conv instead of 7x7 to preserve 64x64 spatial resolution
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, base_width)
        self.relu = nn.ReLU(inplace=True)
        # No initial MaxPool to preserve spatial dimensions

        self.layer1 = self._make_layer(BasicBlock, base_width, 2, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, base_width * 2, 2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, base_width * 4, 2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(BasicBlock, base_width * 8, 2, stride=2, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_width * 8 * BasicBlock.expansion, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        x = self.relu(self.gn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        
        pooled = self.avgpool(f4)
        out = self.fc(pooled)
        if return_spatial:
            return out, [f3, f4]
        return out
