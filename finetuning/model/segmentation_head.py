import torch.nn as nn


class SegmentationHead(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(in_features, in_features // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_features // 2),
            nn.GELU(),
            nn.ConvTranspose2d(in_features // 2, in_features // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_features // 4),
            nn.GELU(),
            nn.ConvTranspose2d(in_features // 4, in_features // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_features // 8),
            nn.GELU(),
            nn.ConvTranspose2d(in_features // 8, in_features // 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_features // 16),
            nn.GELU(),
            nn.Conv2d(in_features // 16, out_features, kernel_size=1)
        )

    def forward(self, x):
        return self.segmentation_head(x)
