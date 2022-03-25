import numpy as np
import torch
import math
import glob
import time
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.create_model import create_model 
import util.util as util
from util.visualizer import Visualizer
import cv2
from train import train_ae_model
from evaluation import evaluation
import shutil
from models import networks

DATASET_DIR = '../mvtec_ad'

def create_dataset(opt, data_root):

    train_rgb_save_path = os.path.join('./datasets', opt.name, 'train')
    test_rgb_save_path = os.path.join('./datasets', opt.name, 'test')

    img_file_list = glob.glob(data_root + '/*.png')
    train_file_list = glob.glob(train_rgb_save_path + '/*.png')
    generate_flag = not (len(train_file_list)==len(img_file_list))

    try:
        if opt.isTrain:
            if generate_flag:
                shutil.rmtree(train_rgb_save_path)
        else:
            shutil.rmtree(test_rgb_save_path)        
    except:
        print('Files do not exist. Creating folders..')

    util.mkdirs([
                train_rgb_save_path, 
                test_rgb_save_path
                ])

    start = time.time()
    if opt.isTrain:
        if generate_flag:
            for img_path in img_file_list:
                img_bgr = cv2.imread(img_path)
                img_bgr = cv2.resize(img_bgr, (opt.loadSize, opt.loadSize), interpolation=cv2.INTER_AREA)

                file_name = img_path[-7:]
                ori_path = os.path.join(train_rgb_save_path, file_name)
                cv2.imwrite(ori_path, img_bgr)
    else:
        for img_path in img_file_list:
            img_bgr = cv2.imread(img_path)
            img_bgr = cv2.resize(img_bgr, (opt.loadSize, opt.loadSize), interpolation=cv2.INTER_AREA)

            file_name = img_path[-7:]
            ori_path = os.path.join(test_rgb_save_path, file_name)
            cv2.imwrite(ori_path, img_bgr) 

    cost = 1000*(time.time()-start)
    print(f'Prepared ...Image preprocessing {cost:.4f}ms')


class Train_and_Valid():
    def __init__(self):
        # Load options and initialize settings
        
        self.validation = False
        self.current_name = None
        self.model_name_list = sorted(glob.glob(DATASET_DIR+'/*'), key=lambda x: x.split('/')[-1])
        self.result_path = './results'
        self.result_dict = OrderedDict()

# training functions ==============================================================
    def train_all(self, target, eval=False):
        for i, path in enumerate(self.model_name_list):
            self.opt = TrainOptions().parse()

            # --- Do not train first n models ------
            # if i < 5:
            #     continue

            name = path.split('/')[-1]
            self.opt.name = name
            self.current_name = name
            self.opt.dataroot = os.path.join('./datasets', self.opt.name)
            train_img_path = os.path.join(DATASET_DIR, name, 'train','good')
            # --- Only train target model ----------
            if name != target and not eval:  # comment this if need to train models of all categories
                continue
            self.train_AE(train_img_path)

    # define your training function. ----------------
    def train_AE(self, train_img_path):
        self.visualizer = Visualizer(self.opt)
        create_dataset(self.opt, train_img_path)
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        train_ae_model(self.opt, dataset, self.visualizer, dataset_size)  # main training function is defined in "train.py"
            

# evaluation functions ============================================================
    def validation_all(self, target, eval=False):
        for i, path in enumerate(self.model_name_list):
            name = path.split('/')[-1]
            if name != target and not eval:
                continue
            self.current_name = name
            self.result_dict[name] = self.validation_single_model()

        if eval:
            AuROC_ma = []
            AuROC_mi = []
            IoU = []
            Precise = []
            Recall = []
            F1 = []
            PRO_AUC = []
            Avg_t = []

            filename = f'Final_result.txt'
            log_save_path = os.path.join(self.result_path, filename)
            t = time.localtime()
            with open(log_save_path, 'a') as f:
                f.write(f'=========== [{t.tm_year}.{t.tm_mon}.{t.tm_mday} // {t.tm_hour} : {t.tm_min}] ===========\n')
                for i, path in enumerate(self.model_name_list):
                    name = path.split('/')[-1]

                    result = self.result_dict[name]
                    AuROC_ma.append(result['AuROC_ma'])
                    AuROC_mi.append(result['AuROC_mi'])
                    Precise.append(result['Precise'])
                    Recall.append(result['Recall'])
                    IoU.append(result['IoU'])
                    F1.append(result['F1'])
                    PRO_AUC.append(result['PRO_AUC'])
                    Avg_t.append(result['Avg_time'])
                    
                    f.write(f'Model: {name}        Avg Time cost: {Avg_t[-1]:.4f}\n')
                    f.write(f'AuROC: [macro]: {AuROC_ma[-1]:.4f}, [micro]: {AuROC_mi[-1]:.4f}     PRO AUC:{PRO_AUC[-1]:.4f}\n')
                    f.write(f'Mean IoU : {IoU[-1]:.4f} Mean Precise : {Precise[-1]:.4f}  Mean Recall : {Recall[-1]:.4f}  Mean F1 : {F1[-1]:.4f} \n')
                    f.write('----------------------\n')

    
                AuROC_ma = np.mean(AuROC_ma)
                AuROC_mi = np.mean(AuROC_mi)
                IoU = np.mean(IoU)
                Precise = np.mean(Precise)
                Recall = np.mean(Recall)
                PRO_AUC = np.mean(PRO_AUC)
                F1 = np.mean(F1)
                Avg_t = np.mean(Avg_t)

                f.write(f'AuROC_ma: {AuROC_ma:.4f}, AuROC_mi: {AuROC_mi:.4f}\n')
                f.write(f'PRO_AUC 30% : {PRO_AUC:.4f}\n')
                f.write(f'IoU: {IoU:.4f}, Precise: {Precise:.4f}\n')
                f.write(f'Recall: {Recall:.4f}, F1_mean: {F1:.4f}\n')
                f.write(f'Avg Process Time: {Avg_t:.4} ms\n\n')
   
    def validation_single_model(self):
        test_path = os.path.join(DATASET_DIR, self.current_name, 'test', '*')
        defect_list = sorted(glob.glob(test_path), key=os.path.getsize)
        self.opt = TestOptions().parse(train=False, save=False)
    
        self.opt.dataroot = os.path.join('./datasets', self.current_name)
        self.opt.name = self.current_name
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.noise_repeat_num = 1   # test code only supports noise_repeat_num = 1
        self.opt.no_flip = True  # no flip

        self.validate_AE(defect_list)

    # define your validation function. --------------
    def validate_AE(self, defect_list):
        # load model
        #  ------------------------   only this method requires load txt -------------
        dim_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'latent_dims.txt')
        n_dim = np.loadtxt(dim_path, dtype=int)
        self.opt.latent_dim = int(n_dim)
        print(f'Latent_dim: {self.opt.latent_dim}')
        # n_dim = 32
        # self.opt.latent_dim = n_dim 
        # ---------------------------------------------------------

        self.opt.model = 'autoencoder'
        AE_model = create_model(self.opt)
        Extractor = networks.Extractor(isTrain=self.opt.isTrain)

        visualizer = Visualizer(self.opt)
        evalu = evaluation(self.opt, self.current_name) 
        time_cost = []

        for idx, path in enumerate(defect_list):
            defect_name = path.split('/')[-1]
            # if defect_name == 'good':
            #     continue
            
            create_dataset(self.opt, path)
            data_loader = CreateDataLoader(self.opt)
            dataset = data_loader.load_data()
            evalu.load_mask_list(defect_name)

            #create save path
            result_dir = os.path.join(self.result_path, self.opt.name)
            save_dir = os.path.join(result_dir, defect_name)
            img_dir = os.path.join(save_dir, 'images')
            util.mkdirs([result_dir, save_dir, img_dir])

            for i, data in enumerate(dataset):
                start_time = time.time()
                features = Extractor(data['input'])[0]

                score_map = AE_model.inference(features)
                score_map = torch.nn.functional.interpolate(score_map, size=(self.opt.loadSize,self.opt.loadSize), mode="bilinear", align_corners=False)
                score = score_map.std().detach().cpu().numpy() 
                time_cost.append(time.time() - start_time)
                visuals = OrderedDict([
                            ('score_map', util.tensor2im(score_map.data[0], normalize=False)),
                            ])
                img_path = data['path']
                visualizer.save_images(img_dir, visuals, img_path)
                if defect_name != 'good':
                    evalu.input_data(util.tensor2im(score_map.data[0], normalize=False))    
                evalu.input_img_score(score)

            if defect_name != 'good':        
                evalu.calculate_single_defect(save_dir)
        
        time_cost = np.mean(time_cost) * 1000
        return evalu.calculate_result(result_dir, time_cost)    



if __name__ == '__main__':

    # ---
    Train_flag = False         # True:  training,  False:  testing
    Eval_all_flag = False      # True:  evaluate all categories     False: only evaluate target category
    target_name = 'bottle'     # Train or evaluate models of target category

    T = Train_and_Valid()
    if Train_flag:
        T.train_all(target_name, Eval_all_flag)
    else:
        T.validation_all(target_name, eval=Eval_all_flag)



