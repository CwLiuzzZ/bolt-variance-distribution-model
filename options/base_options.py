import argparse
import os
from util import util
import torch
import cv2
from collections import OrderedDict

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='bottle', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 3  0,1,2,3 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='texture', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--top_k', type=int, default=5, help='for spade, the num of nearest pic')
        self.parser.add_argument('--kp_num', type=int, default=11, help='num of keypoints for pvnet')
        self.parser.add_argument('--kp_zone_size', type=int, default=1307, help='size of one whole keypointzone')
        self.parser.add_argument('--save_path', type=str, default='./results/spade_visual', help='for spade')
        self.parser.add_argument('--students_save_path', type=str, default='./results/student_visual', help='for students')

        # for students
        self.parser.add_argument('--n_students', type=int, default=3, help="Number of students network to train")
        self.parser.add_argument('--student_type', type=int, default=0, help="network of students 0 means simple conv, 1 means resnet")
        self.parser.add_argument('--stu_id', type=int, default=0, help="which student network")
        self.parser.add_argument('--loss_weight', type=list, default=[3,3,3,4,4],  help='weight when calculate loss and get anomaly')
        self.parser.add_argument('--sum_loss_weight', type=int, default=10,  help='sum of loss_weight')
        # 33,65还不能用，hook没添加，网络�??????间层特征尺�?�也与pvnet不一�??????
        self.parser.add_argument('--patch_size', type=int, default=17, choices=[17, 33, 65], help="Height and width of patch CNN")

        # self.parser.add_argument('--defect_model', action='store_true', default=False, help='choose the anomaly model defect/normal')
        # self.parser.add_argument('--augment_dataset', action='store_true', default=False, help='set the train dataset')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./dataset/') 
        # self.parser.add_argument('--aug_dataroot', type=str, default='./datasets/cityscapes/')  
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')                

        # for displays
        self.parser.add_argument('--display', action='store_true', help='Use visdom display and save images.')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')

        # for generator
        self.parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in first conv layer')
        self.parser.add_argument('--layer', type=int, default=3, help='selects a layer to get features 1-5')
        self.parser.add_argument('--n_downsample_global', type=int, default=6, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=3, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--latent_dim', type=int, default=4, help='num of latent dims')    
        self.parser.add_argument('--keypoint', type=int, default=11, help='number of keypoints')
        # for instance-wise features
        # self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        # self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        # self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        # self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        # self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')

        self.initialized = True

    def parse(self, train=True, save=True, spade = False, students = False):
        if not self.initialized:
            self.initialize()
        
        lst = util.load_args(train, spade, students)
        self.opt = self.parser.parse_args(args=lst)
        # get sum of loss_weight
        weight_sum = 0
        for item in self.opt.loss_weight:
            weight_sum+= int(item)
        lst += ['--sum_loss_weight',str(weight_sum)]
        self.opt = self.parser.parse_args(args=lst)
        
        self.opt.isTrain = self.isTrain   # train or test
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        self.opt.dataroot = os.path.join('./dataset', self.opt.name)
        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.load_pretrain = expr_dir

        util.mkdirs(expr_dir)
        return self.opt
