import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel 
from . import networks
from torchsummary import summary
import util.util as util

class DenoisingModel(BaseModel):
    def name(self):
        return 'DenoisingModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt) 
        self.isTrain = opt.isTrain

        layer_1 = 3 + opt.layer * 3 
        layer_2 = layer_1 + 3
        self.layers = [layer_1, layer_2]    # the feature layer index

        ##### define networks        
        # Generator network
        netG_input_nc = opt.input_nc 
        netG_output_nc = opt.output_nc
        netD_input_nc = netG_input_nc + netG_output_nc

        norm_layer = networks.get_norm_layer(norm_type='instance')  
        self.netG = networks.Generator(netG_input_nc, netG_output_nc, opt.ngf, opt.n_downsample_global, opt.n_blocks_global, norm_layer)
        self.netD = networks.MultiscaleDiscriminator(netD_input_nc, opt.ndf, opt.n_layers_D, norm_layer, use_sigmoid=False, num_D=opt.num_D, getIntermFeat=True)
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG.cuda()
            self.netD.cuda()
        self.netG.apply(networks.weights_init)    
        self.netD.apply(networks.weights_init)

        # print(self.netG)


        # load networks
        if not self.isTrain or opt.continue_train:
            print('----------  Loading models  ----------')
            pretrained_path = None if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', pretrained_path) 
            if self.isTrain:          
                self.load_network(self.netD, 'D', pretrained_path)  

        if self.isTrain:
            self.netG.train()
        else:
            self.netG.eval()

        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionConst = torch.nn.L1Loss()
            self.criterionssim = SSIM()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_GAN_Const', 'G_VGG', 'D_real', 'D_fake']

            # initialize optimizers
            # optimizer G
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))                            
            # optimizer D                           
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, input_tensor, label_tensor):   
        input_tensor = Variable(input_tensor.data.cuda())
        label_tensor = Variable(label_tensor.data.cuda())
        return [input_tensor, label_tensor]

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)
    
    def forward(self, data):
        
        noise_img, real_img = self.encode_input(Variable(data['input']), Variable(data['label']))
        fake_image= self.netG.forward(noise_img)

        loss_G_GAN_Const = (0.8 * (1-self.criterionssim(fake_image, real_img)) + (1-0.8) * self.criterionConst(fake_image, real_img)) * 20

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(noise_img, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)  

        # Real Detection and Loss        
        pred_real = self.discriminate(noise_img, real_img)
        loss_D_real = self.criterionGAN(pred_real, True) 

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((noise_img, fake_image), dim=1))    
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(self.opt.num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = self.criterionVGG(fake_image, real_img) * self.opt.lambda_feat
        
        # Only return the fake_B image if necessary to save BW
        losses = (loss_G_GAN, loss_G_GAN_Const, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake)
        return [ losses, fake_image]
    
    def inference(self, data):
        input_img, _ = self.encode_input(Variable(data['input']), Variable(data['label']))
        with torch.no_grad():
            features, fake_image = self.netG.feature_extraction(input_img, self.layers)
        return features, fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)






















