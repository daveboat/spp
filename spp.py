"""
Spatial pyramid pooling

Module for the spatial pyramid pooling (SPP) module used in classification, object detection, and segmentation tasks in
computer vision. In a general sense, along with techniques like dilated convolutions and other feature pyramid methods,
SPP is said to endow local predictions with regional and global context. For example, in the segmentation case, the
final prediction is made using a stack of feature maps, one globally pooled (1x1), one pooled into 4 regions (2x2), etc.
These are stacked with the regular CNN base's feature maps for final convolution into class logits for prediction. In
this way, each pixel's prediction has direct access to context about the entire image, and its region, as well as local
features. So, a pixel might be more likely to be considered part of a car if it sees that road-like textures are in the
image, or that a tree is nearby.

This module is written to accommodate two styles of feature concatenation:

1. Flattened (1D) concatenation, a la https://arxiv.org/abs/1406.4729
    Here, the resulting feature maps are flattened into (batch, -1) tensors, and then concatenated along dim 1. This
    results in a (batch, sum_i (l_i * l_i * in_channels)) output tensor. This more or less faithfully recreates the SPP
    module in the original spatial pyramid pooling paper.
2. Feature map (2D) concatenation, a la https://arxiv.org/abs/1612.01105
    Here, the resulting feature maps are not flattened, but rather concatenated into (batch, C', H, W) tensors, where C'
    is the sum of the channels in the incoming feature maps, plus all pooled feature maps. This results in a
    (batch, 2 * in_channels - in_channels % levels, H, W) output tensor, where the in_channels % levels difference is
    due to floor division in computing the number of channels each pooled feature map should have, when in_channels is
    not divisible by the number of levels. This more or less faithfully recreates the SPP module in the PSPNet paper.

Note that, in the first case, the function would have no weights, and could therefore doesn't need to be represented as
a Module, but in the second case, convolutions with weights are needed.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, levels=(6, 3, 2, 1), pool_mode='max', concat_mode=0):
        """
        Spatial pyramid pooling module
        :param in_channels: Number of channels in the incoming feature tensor
        :param levels: A list of levels for pooling. Each level, k, divides x into k by k cells, each of which gets
        pooled. So the intermediate output is a collection of k_i x k_i feature maps, for k_i in levels. In both the
        original paper and the PSPNet paper, they use (6x6, 3x3, 2x2, 1x1)
        :param pool_mode: Max pooling or average pooling. In the original paper, they use max pooling, while in the
        PSPNet paper, they use average pooling
        :param concat_mode: 0: 1D concatenation. Returns a (batch, -1) tensor. 1: 2D concatenation. Returns a
        (batch, sum_of_channels, H, W) tensor.
        """
        super().__init__()

        # some assertion checks to get us started
        assert pool_mode in ['avg', 'max'], 'pool_mode must be one of \'avg\' and \'max\''
        assert concat_mode in [0, 1], 'concat_mode must be 0 or 1'

        self.levels = levels
        self.pool_mode = pool_mode
        self.concat_mode = concat_mode

        # grab our pooling function
        self.pool = F.max_pool2d if pool_mode == 'max' else F.avg_pool2d

        # create convolutional layers for 2D concatenation only if concat_mode is 1. In this case, each pooled feature
        # is 1x1 convolved to have in_channels / number_of_levels output channels. I use floor division here in case
        # in_channels is not exactly divisible by number_of_levels
        if concat_mode:
            out_channels = in_channels // len(levels)
            self.conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(len(levels))])

    def forward(self, in_features):
        """
        :param in_features: Input tensor, expected to be in (N, C, H, W) format
        :return: In 1D concatenation mode, returns a (N, K) tensor, where K = sum_i (l_i * l_i * in_channels). In 2D
        concatenation mode, returns a (N, K, H, W) tensor, where K = 2 * in_channels - in_channels % levels. The
        difference of in_channels % levels is due to the fact that I'm using floor division to compute the number of
        channels that each pooled
        """
        # get batch dimension, feature map sizes
        N, H, W = in_features.size(0), in_features.size(2), in_features.size(3)

        ret_list = []

        for i in range(len(self.levels)):
            # As stated in the original paper,
            # "For a pyramid level with nxn bins, the (i, j)-th bin is in the range of
            # [floor((i-1)w/n), ceil((i+1)w/n)] x [floor((j-1)h/n), ceil((j+1)h/n)]"
            # Therefore, in order to guarantee our output size is lxl, for each level, we pool with a kernel and stride
            # (H/l, W/l). We want to make sure we don't miss any pixels, so we use the ceiling of H/l and W/l. This
            # requires the image have at least kernel * l pixels along each dimension, so we zero pad to make up the
            # difference, if there is a difference.
            l = self.levels[i]
            kernel_h = int(math.ceil(H / l))
            kernel_w = int(math.ceil(W / l))
            pad_h = int(math.ceil((kernel_h * l - H) / 2))
            pad_w = int(math.ceil((kernel_w * l - W) / 2))

            pooled = self.pool(in_features, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w),
                               padding=(pad_h, pad_w))

            # if we're doing 1D concatenation, flatten. otherwise, put through 1x1 convolutional layer, then upsample to
            # HxW via bilinear interpolation
            if self.concat_mode:
                pooled = self.conv[i](pooled)
                pooled = F.interpolate(pooled, size=(H, W), mode='bilinear')
                ret_list.append(pooled)
            else:
                ret_list.append(pooled.view(N, -1))

        # Add the incoming feature maps to the output stack
        if self.concat_mode:
            ret_list.append(in_features)

        return torch.cat(ret_list, 1)


if __name__ == '__main__':
    # some quick tests for the SPP module
    C = 10
    x = torch.randn((2, C, 20, 20))

    SPP = SpatialPyramidPooling(C)

    print(SPP(x).size())

    SPP_2D = SpatialPyramidPooling(C, concat_mode=1)
    print(SPP_2D(x).size())
