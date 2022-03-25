import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math

# from einops import reduce


class conv_student(nn.Module):
    ''' pure cnn type of resnet-18
    '''
    def __init__(self):
        super(conv_student, self).__init__()

        self.conv_pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)) 

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True))

    def forward(self, x, fdfe=False):
        self.outputs = []
        x0 = self.conv_pre(x)
        x_ = self.maxpool(x0)
        x1 = self.conv1(x_)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return [x0,x1, x2, x3, x4]

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class res_student(nn.Module):

    def __init__(self,
                 block=BasicBlock,
                 layers=[2,2,2,2],
                 output_stride=8):

        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1

        self.inplanes = 64
        super(res_student, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:

                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:

                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # self.current_dilation = 1
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=self.current_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input: [1, 3, 384, 512]
        x = self.conv1(x)
        # conv1: [1, 64, 192, 256]
        x = self.bn1(x)
        # bn1: [1, 64, 192, 256]
        x2s = self.relu(x)
        # x2s: [1, 64, 192, 256]
        x = self.maxpool(x2s)
        # maxpool: [1, 64, 96, 128]

        x4s = self.layer1(x)
        # x4s: [1, 64, 96, 128]
        x8s = self.layer2(x4s)
        # x8s: [1, 128, 48, 64]
        x16s = self.layer3(x8s)
        # x16s: [1, 256, 48, 64]
        x32s = self.layer4(x16s)
        # x32s: [1, 512, 48, 64]
        x = x32s

        return [x2s, x4s, x8s, x16s, x32s]
