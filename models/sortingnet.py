###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

import torch.nn as nn
import ai8x
import numpy as np
import torch

class SortingClassifier128(nn.Module):
    def __init__(self, num_classes=5, num_channels=3, dimensions=(128, 128), bias=True, **kwargs):
        super(SortingClassifier128,self).__init__()
        
        # 3x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 8, 3, stride=1, padding=1,
                                            bias=True,batchnorm='Affine', **kwargs)
        
        # 8x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv2 = ai8x.FusedConv2dBNReLU(8, 8, 3, stride=1, padding=1,
                                            bias=True,batchnorm='Affine', **kwargs)
        
        # 8x128x128 --> 8x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 16, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine', **kwargs)
        bias=True
        # 16x64x64 --> 16x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        
        
        # 32x32x32 --> 32x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        
        # 64x16x16 --> 64x8x8 --> 64x8x8 (padding by 1 so same dimension)
        self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        
        # 64x8x8 --> 64x4x4 --> 64x4x4 (padding by 1 so same dimension)
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(64*4*4, 10, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(10, 5, bias=True, wide=True, **kwargs)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        self.conv_do = torch.nn.Dropout2d()
        self.lin_do = torch.nn.Dropout()
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.lin_do(x)
        x = self.fc1(x)
        x = self.lin_do(x)
        x= self.fc2(x)
        return x


def sortingnet(pretrained=False, **kwargs):
    """
    Constructs a sorting model.
    """
    assert not pretrained
    return SortingClassifier128(**kwargs)


class SimpleSortingClassifier128(nn.Module):
    def __init__(self, num_classes=5, num_channels=3, dimensions=(128, 128), bias=False, **kwargs):
        super().__init__()
        
        # 3x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = ai8x.FusedConv2dReLU(3, 8, 3, stride=1, padding=1,
                                            bias=False, **kwargs)
        
        # 8x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv2 = ai8x.FusedConv2dReLU(8, 8, 3, stride=1, padding=1,
                                            bias=False, **kwargs)
        
        # 8x128x128 --> 8x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(8, 16, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=False, **kwargs)
        bias=True
        # 16x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv4 = ai8x.FusedConv2dReLU(16, 16, 3, stride=1, padding=1,
                                                   bias=bias, **kwargs)
        
        
        # 16x64x64 --> 16x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(16, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias, **kwargs)
        
        # 32x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv6 = ai8x.FusedConv2dReLU(32, 32, 3, stride=1, padding=1,
                                                   bias=bias, **kwargs)
        
        # 32x32x32 --> 32x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv7 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias,**kwargs)
        
        # 64x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv8 = ai8x.FusedConv2dReLU(64, 64, 3, stride=1, padding=1,
                                                   bias=bias, **kwargs)
        
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        self.conv9 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias,**kwargs)
        
        # 64x8x8 --> 64x4x4 (padding by 1 so same dimension)
        self.conv10 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=bias,**kwargs)
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(64*4*4, 12, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(12, 6, bias=True, wide=True, **kwargs)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


def simplesortingnet(pretrained=False, **kwargs):
    """
    Constructs a sorting model.
    """
    assert not pretrained
    return SimpleSortingClassifier128(**kwargs)


models = [
    {
        'name': 'sortingnet',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'simplesortingnet',
        'min_input': 1,
        'dim': 2,
    },
]