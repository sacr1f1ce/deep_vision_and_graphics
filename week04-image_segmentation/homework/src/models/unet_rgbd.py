import torch
import torch.nn as nn
from models.unet import UNetResNet50

__all__ = ["UNetResNet50RGBD"]


class UNetResNet50RGBD(UNetResNet50):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
        )

        original_conv1 = self.encoder1[0]

        new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        with torch.no_grad():
            rgb_weights = original_conv1.weight.data
            depth_weight = torch.zeros_like(rgb_weights[:, :1, :, :])
            new_conv1.weight.data = torch.cat([rgb_weights, depth_weight], dim=1)

            if original_conv1.bias is not None:
                new_conv1.bias.data = original_conv1.bias.data.clone()

        self.encoder1[0] = new_conv1
