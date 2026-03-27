import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightweight re-implementation of the ViSD-Boost image encoder so we can
# load the visual backbone weights from ViSD-Boost/checkpoint.pth without
# depending on that repository at runtime.


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4), device=out.device
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, shortcut_type="B", in_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = lambda x: downsample_basic_block(x, planes * block.expansion, stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


class ViSDVisionBackbone(nn.Module):
    """ResNet-based visual encoder from ViSD-Boost (lavis.utils.vit.ViT)."""

    def __init__(self, in_channels=1, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.res_model = resnet18(shortcut_type="A", in_channels=in_channels)
        self.proj1 = nn.Conv3d(64, hidden_size, kernel_size=1)
        self.proj2 = nn.Conv3d(128, hidden_size, kernel_size=1)
        self.proj3 = nn.Conv3d(256, hidden_size, kernel_size=1)
        self.proj4 = nn.Conv3d(512, hidden_size, kernel_size=1)

    def forward(self, x):
        # x: [B, C, D, H, W]
        res_x1, res_x2, res_x3, res_x4 = self.res_model(x)
        res_x4 = self.proj4(res_x4)
        res_x4 = res_x4.flatten(2).transpose(1, 2)
        return res_x4


class ViSDVisionTower(nn.Module):
    """
    Wraps the ViSD-Boost visual encoder to integrate with the LaMed vision tower
    interface. Outputs patch tokens shaped [B, N, hidden].
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = ViSDVisionBackbone(in_channels=config.image_channel)
        # ResNet strides: depth /16, height /32, width /32 (given conv+pool chain).
        self.patch_stride = (16, 32, 32)
        self.num_patches = self._infer_num_patches(config.img_size)

    def _infer_num_patches(self, img_size):
        d, h, w = img_size
        sd, sh, sw = self.patch_stride
        # use ceil to be robust to non-divisible shapes
        return int(math.ceil(d / sd) * math.ceil(h / sh) * math.ceil(w / sw))

    def forward(self, images):
        return self.vision_tower(images)

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

