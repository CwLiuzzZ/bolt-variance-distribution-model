import numpy as np
import math
import torch
import os
from torch.autograd import Variable
from .base_model import BaseModel 
from . import networks
from . import pvnet
from . import students_model
import util.global_variables as gl

from torchsummary import summary
import util.util as util
import cv2

class PVModel(BaseModel):
    def name(self):
        return 'PVModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt) 
        self.isTrain = opt.isTrain
        self.opt = opt
        ##### define networks  
        seg_dim = 1
        self.pvnet = pvnet.get_res_pvnet(2*opt.keypoint, seg_dim) 

        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.pvnet.cuda() 
        self.pvnet.apply(networks.weights_init)

        # load networks
        if not self.isTrain or opt.continue_train:
            print('----------  Loading models  ----------')
            pretrained_path = None if not self.isTrain else opt.load_pretrain
            self.load_network(self.pvnet, 'pv', pretrained_path) 

        if self.isTrain:
            self.pvnet.train()
            self.old_lr = opt.lr
            # define loss functions
            self.vote_criterion = torch.nn.SmoothL1Loss()
            self.seg_criterion = torch.nn.BCELoss()
            # Names so we can breakout loss
            self.loss_names = ['seg_loss', 'vote_loss']
            # initialize optimizers
            # optimizer G
            params = list(self.pvnet.parameters())        
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999)) 
        else:
            self.pvnet.eval()                           

    def encode_input(self, input_tensor, mask_tensor=None, vector_tensor=None):   
        input_tensor = Variable(input_tensor.data.cuda())
        if mask_tensor is None:
            return input_tensor
        mask_tensor = Variable(mask_tensor.data.cuda())
        vector_tensor = Variable(vector_tensor.data.cuda())
        return [input_tensor, mask_tensor, vector_tensor]

    def create_kp_map(self, input_kp_map):
        '''
        input_kp_map: tensor  (b, vn_2 //2, 4)
        '''
        if input_kp_map.is_cuda:
            input_kp_map.cpu()

        b, num_kp, _ = input_kp_map.shape
        _, _, h, w = self.shape
        maps = torch.tensor([])
        each_kp_map_list = []
        keypoint_list = []
        for i in range(b):
            bg = np.zeros((h,w), dtype=np.float)
            for ii in range(num_kp):
                x, x_var, y, y_var = input_kp_map[i, ii, :]
                keypoint_list.append(torch.tensor([y,x]))
                if torch.isnan(x):
                    x = 0
                if torch.isnan(y):
                    y = 0
                temp = np.zeros((h,w), dtype=np.float)
                # cv2.rectangle(temp, (int(x)-18,int(y)-18), (int(x)+18,int(y)+18),(10,10,10), -1)
                cv2.ellipse(temp, (int(x),int(y)), (20,20), 0, 0, 360, (10,10,10), -1)
                bg += temp
                temp[temp > 0] = 1
                temp = torch.from_numpy(temp).float()
                each_kp_map_list.append(temp)
            bg[bg > 0] = 1
            bg = torch.from_numpy(bg).unsqueeze(0).unsqueeze(0).float()
            maps = torch.cat([maps, bg], dim = 0)
        each_kp_map = torch.stack(each_kp_map_list)
        keypoint_list = torch.stack(keypoint_list)
        return maps,each_kp_map,keypoint_list

    def forward(self, data):
        input_tensor, mask_tensor, vector_tensor = self.encode_input(data['input'], data['mask'], data['vmap'])
        # print(input_tensor.shape, mask_tensor.shape, vector_tensor.shape)
        self.shape = input_tensor.shape
        output = self.pvnet(input_tensor)

        output_mask = torch.sigmoid(output['seg'])
        seg_loss = self.seg_criterion(output_mask, mask_tensor)*5
        
        # output_vertex = torch.
        # vote_loss = self.vote_criterion(output['vertex']*mask_tensor, vector_tensor*mask_tensor)
        # vote_loss = vote_loss / mask_tensor.sum()
        # vote_residual_map = torch.abs(output['vertex']*mask_tensor - vector_tensor*mask_tensor).sum(dim=1, keepdim=True)
        vote_loss = self.vote_criterion(output['vertex'], vector_tensor) * 1000
        vote_residual_map = torch.abs(output['vertex'] - vector_tensor).sum(dim=1, keepdim=True)
        losses = (seg_loss, vote_loss)
        maps,each_kp_map,keypoint_list = self.create_kp_map(output['kp_map'])
        return [ losses, output_mask, vote_residual_map, maps]
    
    def inference(self, data):
        input_tensor = self.encode_input(data['input'])
        self.shape = input_tensor.shape
        with torch.no_grad():
            output = self.pvnet(input_tensor)
            # output_mask: key zone mask 0/1
            output_mask = torch.sigmoid(output['seg'])
            # maps: draw circle around 11 keypoints map 0/1
            maps,each_kp_map,keypoint_list = self.create_kp_map(output['kp_map'])  
            dis_map = output['dis_map'].cpu() * maps / 100.0
        return output_mask, maps, dis_map, output['feat'], output['vertex'],each_kp_map,keypoint_list

    def inference_upbone(self, data, feature_list):
        [x2s, x4s, x8s, x16s, x32s] = feature_list
        input_tensor = self.encode_input(data['input'])
        self.shape = input_tensor.shape
        with torch.no_grad():
            output = self.pvnet.inference_upbone(input_tensor, x2s, x4s, x8s, x16s, x32s)
            output_mask = torch.sigmoid(output['seg'])
            maps,each_kp_map,keypoint_list = self.create_kp_map(output['kp_map']) 
            dis_map = output['dis_map'].cpu() * maps / 100.0
        return output_mask, maps, dis_map, output['feat'], output['vertex']

    def save(self, which_epoch):
        self.save_network(self.pvnet, 'pv', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr


class StudentModel(BaseModel):
    def name(self):
        return 'StudentModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt) 
        self.isTrain = opt.isTrain
        self.opt = opt
        self.id = opt.stu_id
        ##### define networks  
        
        if opt.student_type == 0:
            print('use simple conv student')
            self.student = students_model.conv_student()
        elif opt.student_type == 1:
            print('use resnet student')
            self.student = students_model.res_student()
            
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.student = self.student.cuda() 
        self.student.apply(networks.weights_init)

        # load networks
        if not self.isTrain or opt.continue_train:
            print('----------  Loading models  ----------')
            pretrained_path = None if not self.isTrain else opt.load_pretrain
            self.load_network(self.student, 'stu'+str(self.id), pretrained_path) 

        if self.isTrain:
            self.student.train()
            self.old_lr = opt.lr

            # define loss functions
            self.criterion = torch.nn.MSELoss()

            # Names so we can breakout loss
            self.loss_names = ['feat_loss']
            # initialize optimizers
            params = list(self.student.parameters())        
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999)) 
        else:
            self.student.eval()                           

    def encode_input(self, input_tensor, mask_tensor=None, vector_tensor=None):   
        input_tensor = Variable(input_tensor.data.cuda())
        if mask_tensor is None:
            return input_tensor
        mask_tensor = Variable(mask_tensor.data.cuda())
        vector_tensor = Variable(vector_tensor.data.cuda())
        return [input_tensor, mask_tensor, vector_tensor]

    def forward(self, data, teacher_feats, kp_maps):
        input_tensor, _, _ = self.encode_input(data['input'], data['mask'], data['vmap'])
        outputs = self.student(input_tensor)
        h, w = input_tensor.shape[-2:]

        loss = 0
        maps = []
        for i, output in enumerate(outputs):
            feat_shape = output.shape
            kp_map = torch.nn.functional.interpolate(kp_maps, size=(feat_shape[-2],feat_shape[-1]), mode="nearest")
            m = 1.0 * (kp_map>-1)
            m += 0.5 * (kp_map>0)
            feat = teacher_feats[i].detach() * m.cuda()
            output = output * m.cuda()
            # feat = teacher_feats[i].detach()
            # output = output
            loss += self.criterion(output, feat) * self.opt.loss_weight[i]
            # residual = torch.pow(output - teacher_feats[i], 2)
            # residual = torch.nn.functional.interpolate(residual, size=(h,w), mode="bilinear", align_corners=False)
            # maps.append(residual)
        loss /= self.opt.sum_loss_weight
        losses = [loss]
        # residual_map = torch.cat(maps, dim=1)
        # residual_map = torch.mean(residual_map, dim=1, keepdim=True)
        # residual_map = torch.clamp(residual_map, 0, 1)
        # residual_map[kp_maps == 0] = 0

        return losses
        # return [ losses, residual_map]
    
    def inference(self, data, teacher_feats, kp_maps):
        input_tensor = self.encode_input(data['input'])
        outputs = self.student(input_tensor)
        h, w = input_tensor.shape[-2:]
        with torch.no_grad():
            maps = []
            for i, output in enumerate(outputs):
                residual = torch.pow(output - teacher_feats[i], 2)
                residual = torch.nn.functional.interpolate(residual, size=(h,w), mode="bilinear", align_corners=False)
                maps.append(residual)
            residual_map = torch.cat(maps, dim=1)
            residual_map = torch.mean(residual_map, dim=1, keepdim=True)
            residual_map = torch.clamp(residual_map, 0, 1)
            residual_map[kp_maps == 0] = 0

        return residual_map

    def get_layer_feature(self,data):
        input_tensor = self.encode_input(data['input'])
        outputs = self.student(input_tensor)
        return outputs

    def save(self, which_epoch):
        self.save_network(self.student, 'stu'+str(self.id), which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr