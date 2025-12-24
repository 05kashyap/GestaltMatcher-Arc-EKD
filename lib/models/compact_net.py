import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DWConvBlock(nn.Module):
    """Depthwise-Separable Convolution Block"""
    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, s, 1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class CompactBranch(nn.Module):
    """A single lightweight branch of the student model"""
    def __init__(self, in_channels=3, emb_dim=512, width=0.5):
        super().__init__()
        c1, c2, c3 = int(64*width), int(128*width), int(256*width)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, 2, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            DWConvBlock(c1, c2, s=2),
            DWConvBlock(c2, c2, s=1),
            DWConvBlock(c2, c3, s=2),
            DWConvBlock(c3, c3, s=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(c3, emb_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x).flatten(1)
        emb = F.normalize(self.fc(x))
        return emb

class CompactEnsemble(nn.Module):
    """The multi-branch student model"""
    def __init__(self, num_classes: int, num_branches: int = 3, in_channels=3, emb_dim=512, width=0.5):
        super().__init__()
        self.branches = nn.ModuleList([CompactBranch(in_channels, emb_dim, width) for _ in range(num_branches)])
        self.classifiers = nn.ModuleList([nn.Linear(emb_dim, num_classes) for _ in range(num_branches)])

    def forward(self, x) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits_list, emb_list = [], []
        for b, c in zip(self.branches, self.classifiers):
            emb = b(x)
            logits = c(emb)
            logits_list.append(logits)
            emb_list.append(emb)
        return logits_list, emb_list