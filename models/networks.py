import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torchsummary import summary
import math

layers = (
    'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4',
    'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
    )

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real, fake_tensor=None):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            if fake_tensor is None:
                create_label = ((self.fake_label_var is None) or
                                (self.fake_label_var.numel() != input.numel()))
                if create_label:
                    fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                    self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            else:
                fake_tensor = torch.nn.functional.interpolate(fake_tensor, input.size())
                self.fake_label_var = Variable(fake_tensor.data, requires_grad=False)

            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, fake_tensor=None):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.layers = ('relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2')
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x, self.layers), self.vgg(y, self.layers)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Networks
##############################################################################

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()        
        activation = nn.ReLU(True)  
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf*mult*2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            dilation = 1
            if i < 3:
                dilation = 2
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, dilation=dilation)]
                
        ### upsample
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model += [nn.Conv2d(ngf * mult, int(ngf * mult * 2), kernel_size=3, stride=1, padding=1),
            #            norm_layer(int(ngf * mult * 2)), activation, nn.PixelShuffle(2)]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]  

        self.model = nn.Sequential(*model)
        
        
    def forward(self, input):
        x = input      
        output = self.model(x)
        return output 
    
    def feature_extraction(self, input, layers):
        x = input
        max_width = 0
        features = torch.Tensor().cuda()
        for i in range(len(self.model)):
            x = self.model[i](x)
            if i in layers:
                width = x.shape[-1]
                max_width = width if width > max_width else max_width
                temp_feat = torch.nn.functional.interpolate(x,(max_width, max_width))
                features = torch.cat([features, temp_feat], dim=1)
        output = x
        return features, output

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, dilation=1):
        super(ResnetBlock, self).__init__()
        self.dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(self.dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(self.dilation)]
        elif padding_type == 'zero':
            p = self.dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=self.dilation),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(self.dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(self.dilation)]
        elif padding_type == 'zero':
            p = self.dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=self.dilation),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-2.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(vgg19, self).__init__()
        features = models.vgg19(pretrained=True).features    # feature layers
        """ vgg.features
        Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace)                                                        # self.relu1_1
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace)                                                        # self.relu1_2
          
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (6): ReLU(inplace)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace)  
              
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (11): ReLU(inplace)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace)
          (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (17): ReLU(inplace)
          
          (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU(inplace)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace)
          (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (24): ReLU(inplace)
          (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): ReLU(inplace)
          
          (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (29): ReLU(inplace)
          (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (31): ReLU(inplace)
          (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (33): ReLU(inplace)
          (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (35): ReLU(inplace)
          
          (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        """
        # hierarchy 1 (level 1)
        self.conv1_1 = features[0]
        self.relu1_1 = features[1]
        self.conv1_2 = features[2]
        self.relu1_2 = features[3]

        # hierarchy 2 (level 2)
        self.pool1 = features[4]
        self.conv2_1 = features[5]
        self.relu2_1 = features[6]
        self.conv2_2 = features[7]
        self.relu2_2 = features[8]

        # hierarchy 3 (level 3)
        self.pool2 = features[9]
        self.conv3_1 = features[10]
        self.relu3_1 = features[11]
        self.conv3_2 = features[12]
        self.relu3_2 = features[13]
        self.conv3_3 = features[14]
        self.relu3_3 = features[15]
        self.conv3_4 = features[16]
        self.relu3_4 = features[17]

        # hierarchy 4 (level 4)
        self.pool3 = features[18]
        self.conv4_1 = features[19]
        self.relu4_1 = features[20]
        self.conv4_2 = features[21]
        self.relu4_2 = features[22]
        self.conv4_3 = features[23]
        self.relu4_3 = features[24]
        self.conv4_4 = features[25]
        self.relu4_4 = features[26]

        # hierarchy 5 (level 5)
        self.pool4 = features[27]
        self.conv5_1 = features[28]
        self.relu5_1 = features[29]
        self.conv5_2 = features[30]
        self.relu5_2 = features[31]
        self.conv5_3 = features[32]
        self.relu5_3 = features[33]
        self.conv5_4 = features[34]
        self.relu5_4 = features[35]

        self.pool5 = features[36]

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, feature_layers):
        # level 1
        conv1_1 = self.conv1_1(x)
        relu1_1 = self.relu1_1(conv1_1)
        conv1_2 = self.conv1_2(relu1_1)
        relu1_2 = self.relu1_2(conv1_2)
        pool1 = self.pool1(relu1_2)

        # level 2
        conv2_1 = self.conv2_1(pool1)
        relu2_1 = self.relu2_1(conv2_1)
        conv2_2 = self.conv2_2(relu2_1)
        relu2_2 = self.relu2_2(conv2_2)
        pool2 = self.pool2(relu2_2)

        # level 3
        conv3_1 = self.conv3_1(pool2)
        relu3_1 = self.relu3_1(conv3_1)
        conv3_2 = self.conv3_2(relu3_1)
        relu3_2 = self.relu3_2(conv3_2)
        conv3_3 = self.conv3_3(relu3_2)
        relu3_3 = self.relu3_3(conv3_3)
        conv3_4 = self.conv3_4(relu3_3)
        relu3_4 = self.relu3_4(conv3_4)
        pool3 = self.pool3(relu3_4)

        # level 4
        conv4_1 = self.conv4_1(pool3)
        relu4_1 = self.relu4_1(conv4_1)
        conv4_2 = self.conv4_2(relu4_1)
        relu4_2 = self.relu4_2(conv4_2)
        conv4_3 = self.conv4_3(relu4_2)
        relu4_3 = self.relu4_3(conv4_3)
        conv4_4 = self.conv4_4(relu4_3)
        relu4_4 = self.relu4_4(conv4_4)
        pool4 = self.pool4(relu4_4)

        # level 5
        conv5_1 = self.conv5_1(pool4)
        relu5_1 = self.relu5_1(conv5_1)
        conv5_2 = self.conv5_2(relu5_1)
        relu5_2 = self.relu5_2(conv5_2)
        conv5_3 = self.conv5_3(relu5_2)
        relu5_3 = self.relu5_3(conv5_3)
        conv5_4 = self.conv5_4(relu5_3)
        relu5_4 = self.relu5_4(conv5_4)
        # pool5 = self.pool5(relu5_4)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return[ out[key] for key in feature_layers ]

class Extractor(nn.Module):
    r"""
    Build muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(self, backbone='vgg19',
                 cnn_layers=layers,
                 upsample="nearest",
                 device='cuda:0',
                 isTrain=False
                ):

        super(Extractor, self).__init__()
        self.device = torch.device(device)
        self.feature = vgg19(requires_grad=False)    # build backbone net
        self.feat_layers = cnn_layers
        if torch.cuda.is_available():
            self.feature.cuda()
        self.feature.eval()

    def forward(self, input_tensor):
        input_tensor = Variable(input_tensor.data.cuda())
        feat_maps = self.feature(input_tensor, feature_layers=self.feat_layers)
        output = []
        features = torch.Tensor().to(self.device)
        for idx, feat_map in enumerate(feat_maps):
            features = torch.cat([features, feat_map], dim=1)
            if (idx+1) % 4 == 0:
                output.append(features)
                features = torch.Tensor().to(self.device)        
        return output

class UnetAE(nn.Module):
    def __init__(self, in_channels=1000, latent_dim=50):
        super(UnetAE, self).__init__()

        self.down_1 = nn.Sequential(
                    nn.Conv2d(in_channels, 8*latent_dim, kernel_size=1, stride=1, padding=0), 
                    nn.InstanceNorm2d(8*latent_dim),
                    nn.ReLU(True))
        self.down_2 = nn.Sequential(
                    nn.Conv2d(8*latent_dim, 4*latent_dim, kernel_size=1, stride=1, padding=0), 
                    nn.InstanceNorm2d(4*latent_dim),
                    nn.ReLU(True))
        self.down_3 = nn.Sequential(
                    nn.Conv2d(4*latent_dim, 2*latent_dim, kernel_size=1, stride=1, padding=0), 
                    nn.InstanceNorm2d(2*latent_dim),
                    nn.ReLU(True))
        self.down_4 = nn.Sequential(
                    nn.Conv2d(2*latent_dim, latent_dim, kernel_size=1, stride=1, padding=0), 
                    nn.InstanceNorm2d(latent_dim),
                    nn.ReLU(True))

        self.up_1 = nn.Sequential(
                    nn.Conv2d(latent_dim, 2*latent_dim, kernel_size=1, stride=1, padding=0), 
                    nn.InstanceNorm2d(2*latent_dim),
                    nn.ReLU(True))
        self.up_2 = nn.Sequential(
                    nn.Conv2d(4*latent_dim, 4*latent_dim, kernel_size=1, stride=1, padding=0), 
                    nn.InstanceNorm2d(4*latent_dim),
                    nn.ReLU(True))
        self.up_3 = nn.Sequential(
                    nn.Conv2d(8*latent_dim, 8*latent_dim, kernel_size=1, stride=1, padding=0), 
                    nn.InstanceNorm2d(8*latent_dim),
                    nn.ReLU(True))
        self.up_4 = nn.Conv2d(16*latent_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_1 = self.down_1(x)
        x_2 = self.down_2(x_1)
        x_3 = self.down_3(x_2)
        x_4 = self.down_4(x_3)
        o_1 = self.up_1(x_4)
        o_2 = self.up_2(torch.cat([o_1, x_3], dim=1))
        o_3 = self.up_3(torch.cat([o_2, x_2], dim=1))
        output = self.up_4(torch.cat([o_3, x_1], dim=1))
        return output

class FRNet(nn.Module):
    def __init__(self, in_channels, latent_dim = 128, use_dropout=False):
        super(FRNet, self).__init__()
        self.in_channels = in_channels
        self.unet = UnetAE(in_channels, latent_dim)

        # self.autoencoder = nn.Sequential(*layers)
    def loss_function(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2)
        return loss

    def compute_energy(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2, dim=0, keepdim=True)
        return loss

    def forward(self, input):
        # unet ==================
        restored = self.unet(input)
        return restored 












