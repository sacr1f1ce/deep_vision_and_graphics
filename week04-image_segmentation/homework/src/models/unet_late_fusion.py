import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNetResNet50

__all__ = ["UNetResNet50LateFusion"]


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.act(out)
        return out


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        mid_channels: int = 8,
        num_layers: int = 1,
        num_resblocks: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=7,
                padding=3,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = mid_channels * (2**i)
            self.encoders.append(
                nn.Sequential(*[ResBlock(in_ch) for _ in range(num_resblocks)])
            )
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        in_ch * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_ch * 2),
                    nn.ReLU(inplace=True),
                )
            )

        bot_ch = mid_channels * (2**num_layers)
        self.bottleneck = nn.Sequential(
            *[ResBlock(bot_ch) for _ in range(num_resblocks)]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(num_layers, 0, -1):
            in_ch = mid_channels * (2**i)
            out_ch = mid_channels * (2 ** (i - 1))

            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    *[ResBlock(out_ch) for _ in range(num_resblocks)],
                )
            )

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        skips = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.downs[i](x)

        x = self.bottleneck(x)

        for i in range(self.num_layers):
            skip = skips.pop()
            x = self.ups[i](x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        x = self.final_up(x)

        return self.final(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)[:, 1]

        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class UNetResNet50LateFusion(UNetResNet50):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        depth_mid_channels: int = 16,
        depth_num_layers: int = 4,
        depth_out_channels: int = 4,
        depth_num_resblocks: int = 1,
        final_channels: int = 32,
    ) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)

        self.dice_loss = DiceLoss()

        self.depth_net = DepthNet(
            in_channels=1,
            out_channels=depth_out_channels,
            mid_channels=depth_mid_channels,
            num_layers=depth_num_layers,
            num_resblocks=depth_num_resblocks,
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                64 + depth_out_channels, final_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, num_classes, kernel_size=1),
        )

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = super().compute_loss(output, target)
        # dice_loss = self.dice_loss(output, target)
        return ce_loss

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_size = inputs.shape[2:]

        rgb = inputs[:, :3]
        depth = inputs[:, 3:]

        rgb_feat = super().forward(rgb, return_features=True)

        depth_feat = self.depth_net(depth)

        fused_feat = torch.cat([rgb_feat, depth_feat], dim=1)

        logits = self.final(fused_feat)

        output = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        return output
