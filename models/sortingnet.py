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
import distiller.apputils as apputils
from train import update_old_model_params

# class SortingClassifier128(nn.Module):
#     def __init__(self, num_classes=5, num_channels=3, dimensions=(128, 128), bias=True, **kwargs):
#         super(SortingClassifier128,self).__init__()
        
#         # 3x128x128 --> 8x128x128 (padding by 1 so same dimension)
#         self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 8, 3, stride=1, padding=1,
#                                             bias=True,batchnorm='Affine', **kwargs)
        
#         # 8x128x128 --> 8x128x128 (padding by 1 so same dimension)
#         self.conv2 = ai8x.FusedConv2dBNReLU(8, 8, 3, stride=1, padding=1,
#                                             bias=True,batchnorm='Affine', **kwargs)
        
#         # 8x128x128 --> 8x64x64 --> 16x64x64 (padding by 1 so same dimension)
#         self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 16, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
#                                                    bias=True,batchnorm='Affine', **kwargs)
#         bias=True
#         # 16x64x64 --> 16x32x32 --> 32x32x32 (padding by 1 so same dimension)
#         self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
#                                                    bias=bias, batchnorm='Affine', **kwargs)
        
        
#         # 32x32x32 --> 32x16x16 --> 64x16x16 (padding by 1 so same dimension)
#         self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
#                                                    bias=bias, batchnorm='Affine', **kwargs)
        
#         # 64x16x16 --> 64x8x8 --> 64x8x8 (padding by 1 so same dimension)
#         self.conv6 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
#                                                    bias=bias, batchnorm='Affine', **kwargs)
        
#         # 64x8x8 --> 64x4x4 --> 64x4x4 (padding by 1 so same dimension)
#         self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
#                                                    bias=bias, batchnorm='Affine', **kwargs)
        
#         # flatten to fully connected layer
#         self.fc1 = ai8x.FusedLinearReLU(64*4*4, 10, bias=True, **kwargs)
#         self.fc2 = ai8x.Linear(10, 5, bias=True, wide=True, **kwargs)

#         # initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
                
#         self.conv_do = torch.nn.Dropout2d()
#         self.lin_do = torch.nn.Dropout()
                
#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = x.view(x.size(0), -1)
#         x = self.lin_do(x)
#         x = self.fc1(x)
#         x = self.lin_do(x)
#         x= self.fc2(x)
#         return x


# def sortingnet(pretrained=False, **kwargs):
#     """
#     Constructs a sorting model.
#     """
#     assert not pretrained
#     return SortingClassifier128(**kwargs)


'''
baseline model (VGG16-like), no batch norm, no dropout, no residual connections
'''
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


'''
adds batchnorm to simple sorting net
'''
class SimpleSortingClassifierBN128(nn.Module):
    def __init__(self, num_classes=5, num_channels=3, dimensions=(128, 128), bias=False, **kwargs):
        super().__init__()
        
        # 3x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = ai8x.FusedConv2dBNReLU(3, 8, 3, stride=1, padding=1,
                                            bias=True,batchnorm='Affine', **kwargs)
        
        # 8x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv2 = ai8x.FusedConv2dBNReLU(8, 8, 3, stride=1, padding=1,
                                            bias=True,batchnorm='Affine', **kwargs)
        
        # 8x128x128 --> 8x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 16, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine', **kwargs)
        bias=True
        # 16x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv4 = ai8x.FusedConv2dBNReLU(16, 16, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        
        # 16x64x64 --> 16x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv6 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # 64x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv8 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # 64x8x8 --> 64x4x4 (padding by 1 so same dimension)
        self.conv10 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
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


def simplesortingnetbn(pretrained=False, **kwargs):
    """
    Constructs a sorting model.
    """
    assert not pretrained
    return SimpleSortingClassifierBN128(**kwargs)


'''
uses embedding for simple sorting net
'''
class SimpleSortingEmb(nn.Module):
    def __init__(self, num_classes=5, num_channels=3, dimensions=(128, 128), bias=False, **kwargs):
        super().__init__()
        
        # 3x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = ai8x.FusedConv2dBNReLU(3, 8, 3, stride=1, padding=1,
                                            bias=True,batchnorm='Affine', **kwargs)
        
        # 8x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv2 = ai8x.FusedConv2dBNReLU(8, 8, 3, stride=1, padding=1,
                                            bias=True,batchnorm='Affine', **kwargs)
        
        # 8x128x128 --> 8x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 16, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine', **kwargs)
        bias=True
        # 16x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv4 = ai8x.FusedConv2dBNReLU(16, 16, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        
        # 16x64x64 --> 16x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv6 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # 64x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv8 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # 64x8x8 --> 64x4x4 (padding by 1 so same dimension)
        self.conv10 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(64*4*4, 64, bias=True, **kwargs)

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
        
        return x


def simplesortingnetemb(pretrained=False, **kwargs):
    """
    Constructs a sorting model.
    """
    assert not pretrained
    return SimpleSortingEmb(**kwargs)


'''
adds bb to simplesortingnetbn
'''
class SimpleSortingClassifierBNBB128(nn.Module):
    def __init__(self, num_classes=5, num_channels=3, dimensions=(128, 128), bias=False, **kwargs):
        super().__init__()
        pretrained_path = "/home/geffen/Documents/ScrapSort/src/ai8x-synthesis/trained/simplesort8_qat.pth.tar"
        
        self.feature_extractor = SimpleSortingClassifierBN128(**kwargs)
        update_old_model_params(pretrained_path, self.feature_extractor)
        ai8x.fuse_bn_layers(self.feature_extractor)
        
        model = apputils.load_lean_checkpoint(self.feature_extractor, pretrained_path)
        ai8x.update_model(model)
        self.feature_extractor = model
        
        # freeze the weights except for last conv and fc
        ct = 0
        for child in self.feature_extractor.children():
            ct += 1
            if ct < 10:
                for param in child.parameters():
                    param.requires_grad = False
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
            
        # retrain the last layer to detect a bounding box and classes
        self.feature_extractor.fc1 = ai8x.Linear(64*4*4, 10, bias=False, wide=True, **kwargs)
            
        # add a fully connected layer for bounding box detection after the conv10
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.conv2(x)
        x = self.feature_extractor.conv3(x)
        x = self.feature_extractor.conv4(x)
        x = self.feature_extractor.conv5(x)
        x = self.feature_extractor.conv6(x)
        x = self.feature_extractor.conv7(x)
        x = self.feature_extractor.conv8(x)
        x = self.feature_extractor.conv9(x)
        x = self.feature_extractor.conv10(x)
        x = x.view(x.size(0), -1)
        
        # output layers
        x1 = self.feature_extractor.fc1(x)

        return x1


def simplesortingnetbnbb(pretrained=False, **kwargs):
    """
    Constructs a sorting model.
    """
    assert not pretrained
    return SimpleSortingClassifierBNBB128(**kwargs)

models = [
    # {
    #     'name': 'sortingnet',
    #     'min_input': 1,
    #     'dim': 2,
    # },
    {
        'name': 'simplesortingnet',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'simplesortingnetemb',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'simplesortingnetbn',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'simplesortingnetbnbb',
        'min_input': 1,
        'dim': 2,
    }
]