import torch
import torch.nn as nn
import torchvision.models as models
from config import cfg

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))

        pt0 = xs_pos * y
        pt1 = xs_neg * (1.0 - y)
        pt = pt0 + pt1

        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
        one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)

        loss = -one_sided_w * (los_pos + los_neg)
        return loss.sum()

class KneeAbnormalityModel(nn.Module):
    def __init__(self, num_classes: int = cfg.num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, slices, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * slices, channels, h, w)
        
        features = self.backbone(x)
        features = features.view(batch_size, slices, -1)
        pooled_features = torch.mean(features, dim=1)
        
        logits = self.head(pooled_features)
        return logits