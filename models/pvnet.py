from torch import nn
import torch
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import util.util as util
import numpy as np
import math
import os


# ===========================================================\
#         ResNet 18 --
# ===========================================================\

# __all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)

        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=False,
                 output_stride=32
    ):

        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1

        self.remove_avg_pool_layer = remove_avg_pool_layer

        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
            # In the latest unstable torch 4.0 the tensor.copy_
            # method was changed and doesn't work as it used to be
            # self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

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

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)
        # avgpool: [1, 512, 48, 64]

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        xfc = self.fc(x)
        # xfc: [1, 256, 48, 64]
        return x2s, x4s, x8s, x16s, x32s, xfc

    def inference_upbone(self,x32s):
        x = x32s

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)
        # avgpool: [1, 512, 48, 64]

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        xfc = self.fc(x)
        # xfc: [1, 256, 48, 64]
        return xfc

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



#  ==============================================================\
#  =====      PVNET      ResNet 18   ============================\
#  ==============================================================\

class Resnet18(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True
                               )

        self.ver_dim=ver_dim
        self.seg_dim=seg_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, raw_dim, 3, 1, 1, bias=False),
            nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def keypoint_map(self, output):
        vertex = output['vertex']
        # [1, 22, 384, 512]
        b, vn_2, h, w = vertex.shape
        mask = torch.sigmoid(output['seg'])
        row = np.arange(h)
        row = row.reshape(h,1).repeat(w, axis=1)
        col = np.arange(w)
        col = col.reshape(1,w).repeat(h, axis=0)
        y_temp = row.astype(np.float32)
        x_temp = col.astype(np.float32)
        y_temp = torch.from_numpy(y_temp).cuda()
        x_temp = torch.from_numpy(x_temp).cuda()

        kp_map = torch.zeros([b,vn_2//2,4])
        dis_map = torch.tensor([]).cuda()
        for bi in range(b):
            mask_bi = mask[bi,0,:,:]
            dis_map_temp = torch.tensor([]).cuda()
            for i in range(vn_2 // 2):
                v_map_x = vertex[bi,2*i+1,:,:]
                v_map_y = vertex[bi,2*i,:,:]
                v_map_x = x_temp - v_map_x * w
                v_map_y = y_temp - v_map_y * h
                v_map_x_ = v_map_x[mask_bi>0.5]
                v_map_y_ = v_map_y[mask_bi>0.5]
                kp_map[bi, i, 0] = torch.mean(v_map_x_)
                kp_map[bi, i, 1] = torch.std(v_map_x_)
                kp_map[bi, i, 2] = torch.mean(v_map_y_)
                kp_map[bi, i, 3] = torch.std(v_map_y_)

                temp_x = torch.abs(v_map_x - torch.mean(v_map_x_)).unsqueeze(0).unsqueeze(0)
                temp_y = torch.abs(v_map_y - torch.mean(v_map_y_)).unsqueeze(0).unsqueeze(0)
                dis = torch.sqrt(torch.pow(temp_x,2) + torch.pow(temp_y,2))
                dis_map_temp = torch.cat([dis_map_temp, dis], dim=1)
            dis_map_temp = torch.sum(dis_map_temp, dim=1, keepdim=True)
            
            # mask_bi = mask_bi.unsqueeze(0).unsqueeze(0)
            # dis_map_temp *= mask_bi
            # dis_map_temp /= 100
            # print(torch.min(dis_map_temp), torch.max(dis_map_temp))
            
            dis_map = torch.cat([dis_map, dis_map_temp], axis=0)

        # dis_map: error of each pixel's prediction of keypoints
        # kp_map: x,y axis of 11 keypoints
        output.update({'kp_map': kp_map, 'dis_map': dis_map})


    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)
        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)
        # if fm.shape[2]==136:
        #     fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,x],1))
        seg_pred=x[:,:self.seg_dim,:,:]
        ver_pred=x[:,self.seg_dim:,:,:]

        # vertex: 22 dimension keypoint location
        ret = {'seg': seg_pred, 'vertex': ver_pred, 'feat': [x2s, x4s, x8s, x16s, x32s]}
        self.keypoint_map(ret)

        # if not self.training:
        #     with torch.no_grad():
        #         self.keypoint_map(ret)

        return ret

    def inference_upbone(self,x,x2s, x4s, x8s, x16s, x32s):
        xfc = self.resnet18_8s.inference_upbone(x32s)
        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)
        # if fm.shape[2]==136:
        #     fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)
        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)
        # fm [1, 64, 192, 256]

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)
        # fm [1, 32, 384, 512]

        x=self.convraw(torch.cat([fm,x],1))
        # x [1, 23, 384, 512]
        seg_pred=x[:,:self.seg_dim,:,:]
        ver_pred=x[:,self.seg_dim:,:,:]

        ret = {'seg': seg_pred, 'vertex': ver_pred, 'feat': [x2s,x4s, x8s, x16s, x32s,xfc]}
        self.keypoint_map(ret)

        # if not self.training:
        #     with torch.no_grad():
        #         self.keypoint_map(ret)
        return ret

def get_res_pvnet(ver_dim, seg_dim):
    model = Resnet18(ver_dim, seg_dim)
    return model











