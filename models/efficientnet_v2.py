

import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

# import pytorch_lightning as pl
# import pytorch_lightning.metrics.functional as plm


# Main Module for efficientnet v2
class EfficientNetV2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # First Convolution
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.act1 = nn.SiLU()

        # Make Blocks Dictionary
        block_dict = OrderedDict()

        # Layer Info : [conv_type, layers, in_channels, out_channels, expansion_ratio, kernel_size, stride]
        for stage, info in enumerate(kwargs['layers_info'], 1) :
            ConvBlock = FusedMBConvBlock if info[0] == 'FusedMBConv' else MBConvBlock

            # Repeat Layers
            for i in range(1, info[1]+1):
                # Stride, inchannels apply to the first layer of each stage
                stride, in_channels = (info[6], info[2]) if i == 1 else (1, info[3])
                block_dict[f'stage_{stage}_{i:02d}'] = ConvBlock(in_channels, *info[3:-1], stride)

        # Fused MBConv Blocks
        self.blocks = nn.Sequential(block_dict)

        # Last Convolution
        last_channel_size = kwargs['layers_info'][-1][3]
        self.conv2 = nn.Conv2d(last_channel_size, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.act2 = nn.SiLU()

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, kwargs['num_classes'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Traditional MBConv Block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, kernel_size, stride):
        super().__init__()

        block_dict = OrderedDict()

        # Expansion Convolution
        hidden_channels = in_channels * expansion_ratio
        block_dict['conv_1'] = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        block_dict['bn_1'] = nn.BatchNorm2d(hidden_channels)
        block_dict['act_1'] = nn.SiLU()

        # Depthwise Convolution
        block_dict['conv_2'] = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding=kernel_size//2, groups=hidden_channels, bias=False)
        block_dict['bn_2'] = nn.BatchNorm2d(hidden_channels)
        block_dict['act_2'] = nn.SiLU()

        # SE Block
        block_dict['se'] = SEBlock(hidden_channels)

        # Projection Convolution
        block_dict['conv_3'] = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        block_dict['bn_3'] = nn.BatchNorm2d(out_channels)


        # Make Block
        self.block = nn.Sequential(block_dict)

        # Identity Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else : 
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.silu(out)
        return out

# Fused MB Convolution Block
class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, kernel_size, stride):
        super().__init__()

        block_dict = OrderedDict()

        # Fused Convolutions
        hidden_channels = in_channels * expansion_ratio
        block_dict['conv_1'] = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        block_dict['bn_1'] = nn.BatchNorm2d(hidden_channels)
        block_dict['act_1'] = nn.SiLU()

        # SE Block
        block_dict['se'] = SEBlock(hidden_channels)

        # Projection Convolution
        block_dict['conv_2'] = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        block_dict['bn_2'] = nn.BatchNorm2d(out_channels)

        # Make Block
        self.block = nn.Sequential(block_dict)

        # Identity Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else : 
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = F.silu(out)
        return out

# Squeeze and Excitation Block        
class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC Layers
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.SiLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.avgpool(x)
        out = out.flatten(1)
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = x * out
        return out



def make_efficientnetv2(model_type:str, num_classes:int):
    if model_type == 's':
        efficientnetv2_s = [
            # [conv_type,       layers, in_channels,    out_channels,   expansion_ratio,    kernel_size,    stride]
            ['FusedMBConv',     2,      24,             24,             1,                  3,              1],
            ['FusedMBConv',     4,      24,             48,             4,                  3,              2],
            ['FusedMBConv',     4,      48,             64,             4,                  3,              2],
            ['MBConv',          6,      64,             128,            4,                  3,              2],
            ['MBConv',          9,      128,            160,            6,                  3,              1],
            ['MBConv',          15,     160,            256,            6,                  3,              2],
        ]
    else :
        raise NotImplementedError('EfficientNetV2-{} is not implemented.'.format(model_type))
    return EfficientNetV2(layers_info = efficientnetv2_s, num_classes = num_classes)


if __name__ == '__main__':
    
    from torchsummary import summary

    efficientnetv2_s = [
        # [conv_type,       layers, in_channels,    out_channels,   expansion_ratio,    kernel_size,    stride]
        ['FusedMBConv',     2,      24,             24,             1,                  3,              1],
        ['FusedMBConv',     4,      24,             48,             4,                  3,              2],
        ['FusedMBConv',     4,      48,             64,             4,                  3,              2],
        ['MBConv',          6,      64,             128,            4,                  3,              2],
        ['MBConv',          9,      128,            160,            6,                  3,              1],
        ['MBConv',          15,     160,            256,            6,                  3,              2],
    ]

    model = EfficientNetV2(layers_info = efficientnetv2_s, num_classes = 10)
    # print(model)

    print(summary(model, ( 3, 224, 224), device = 'cpu'))





