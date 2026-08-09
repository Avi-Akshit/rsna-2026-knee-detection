import torch
import torch.nn as nn
import timm
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

class RSNA2D5Model(nn.Module):
    def __init__(
        self, 
        model_name: str = cfg.model_name, 
        pretrained: bool = cfg.pretrained, 
        in_channels: int = cfg.in_channels, 
        num_classes: int = cfg.num_classes
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate
        )
        
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits