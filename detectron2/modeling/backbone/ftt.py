import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron.layers import Conv2d, ShapeSpec, get_norm

import math

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone


# p2, p3 in the paper is p3, p4 for us 
# format of p2, p3 is both [bs, channels, height, width]
#capsule detection
def FTT_get_p3pr(p2, p3, p4,p5,out_channels, norm):
    channel_scaler = Conv2d(
        out_channels,
        out_channels * 4,
        kernel_size=1,
        bias=False
        #norm=''
    )

    # tuple of (conv2d, conv2d, iter)  FTT
    def create_content_extractor(x, num_channels, iterations=3):
        conv1 = Conv2d(
        num_channels,
        num_channels,
        kernel_size=1,
        bias=False,
        #norm=get_norm(norm, num_channels),
        )

        conv2 = Conv2d(
        num_channels,
        num_channels,
        kernel_size=1,
        bias=False,
        #norm=get_norm(norm, num_channels),
        )

        out = x
        for i in range(iterations):
            out = conv1(out)
            out = F.relu_(out)
            out = conv2(out)
            out = F.relu_(out)

        return out

    def create_texture_extractor(x, num_channels, iterations=3):
        conv1 = Conv2d(
        num_channels,
        num_channels,
        kernel_size=1,
        bias=False,
        #norm=get_norm(norm, num_channels),
        )

        conv2 = Conv2d(
        num_channels,
        num_channels,
        kernel_size=1,
        bias=False,
        #norm=get_norm(norm, num_channels),
        )

        conv3 = Conv2d(
        num_channels,
        int(num_channels/2),
        kernel_size=1,
        bias=False,
        )

        out = x
        for i in range(iterations):
            out = conv1(out)
            out = F.relu_(out)
            out = conv2(out)
            out = F.relu_(out)
        out = conv3(out)
        out = F.relu_(out)
        return out

    bottom_1=p3
    bottom_1 = channel_scaler(bottom_1)
    bottom_1 = create_content_extractor(bottom_1, out_channels*4)
    sub_pixel_conv = nn.PixelShuffle(2)
    bottom_1 = sub_pixel_conv(bottom_1)
    #print("\np3 shape: ",bottom.shape,"\n")

    bottom_2 = p4
    bottom_2 = channel_scaler(bottom_2)
    bottom_2 = create_content_extractor(bottom_2, out_channels*4)
    sub_pixel_conv = nn.PixelShuffle(2)
    bottom_2 = sub_pixel_conv(bottom_2)

    bottom_3 = p5
    bottom_3 = channel_scaler(bottom_3)
    bottom_3 = create_content_extractor(bottom_3, out_channels*4)
    sub_pixel_conv = nn.PixelShuffle(2)
    bottom_3 = sub_pixel_conv(bottom_3)

    bottom = bottom_1+bottom_2 + bottom_3


    # We interpreted "wrap" as concatenating bottom and top
    # so the total channels is doubled after (basically place one on top
    # of the other)
    top = p2
    top = torch.cat((bottom, top), axis=1)
    top = create_texture_extractor(top, out_channels*2)
    #top = top[:,256:]

    result = bottom + top

    return result
