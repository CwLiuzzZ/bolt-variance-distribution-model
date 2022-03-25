import numpy as np
import torch
import math
import glob
import time
import os
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from skimage import measure
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.visualizer import Visualizer
import cv2
import collections

DATASET_DIR = '../mvtec_ad'

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def per_region_overlap(score, mask, expect_fpr=0.3):
    # per region overlap
    score = np.array(score)
    mask = np.array(mask)

    max_step = 50
    max_th = score.max()
    min_th = score.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        #     print(thred)
        # segmentation
        binary_score_maps[score <= thred] = 0
        binary_score_maps[score > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = measure.label(mask[i], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[i])
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr
        masks_neg = ~mask
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # save results
    # data = np.vstack([threds, fprs, pros_mean, pros_std])
    # df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
    #                                                 'pros_mean', 'pros_std'])
    # df_metrics.to_csv(os.path.join(self.eval_path, '{}_pro_fpr.csv'.format(self.model_name)),
    #                   sep=',', index=False)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr    # # rescale fpr [0,0.3] -> [0, 1]
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])    # need scale
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr*100)), pro_auc_score)

    return pro_auc_score

class evaluation():
    def __init__(self, opt, model_name):
        self.opt = opt
        self.model_name = model_name       
        self.defect_mask_list = []
        self.img_score_list = []
        # self.uncertainty_score_list = []
        self.img_label_list = []
        self.FPR = {}
        self.TPR = {}
        self.IoU = {}
        self.Precise = {}
        self.Recall = {}
        self.F1 = {}
        self.AuROC = {}

        self.scores = []
        self.masks = []

    def load_mask_list(self, defect_name):
        self.defect_mask_list = []
        self.defect_name = defect_name
        self.gt_file_path = os.path.join(DATASET_DIR, self.model_name, 'ground_truth', defect_name)
        self.img_file_path = os.path.join(DATASET_DIR, self.model_name, 'test', defect_name)
        self.gt_mask_list, self.gt_img_list, img_level_label = self.load_gt_imgs()
        self.img_label_list.extend(img_level_label)
        print(f'Current Model:{self.model_name}, Current Defect:{defect_name} \nGT_images loaded... Length:{len(self.gt_mask_list)}')

    def load_gt_imgs(self):
        mask_list = []
        img_list = []
        label_list = []
        if self.defect_name != 'good':
            file_path = glob.glob(os.path.join(self.gt_file_path, '*.png'))
            file_path.sort(key=lambda x:int(x[-12:-9]))
            img_path = glob.glob(os.path.join(self.img_file_path, '*.png'))
            img_path.sort(key=lambda x:int(x[-7:-4]))
            for idx, path in enumerate(file_path):
                gt = cv2.imread(path)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                gt = cv2.resize(gt, (self.opt.loadSize, self.opt.loadSize))
                img = cv2.imread(img_path[idx])
                img = cv2.resize(img, (self.opt.loadSize, self.opt.loadSize))
                mask_list.append(gt)
                img_list.append(img)
                label_list.append(np.ones(1))
        else:
            file_path = glob.glob(os.path.join(self.img_file_path, '*.png'))
            label_list.extend([np.zeros(1)]*len(file_path))
            mask = np.ones((256, 256), dtype=np.uint8)
            mask_list.append(mask)
            img = np.ones((256, 256, 3), dtype=np.uint8)
            img_list.append(mask)
        return mask_list, img_list, label_list

    def input_data(self, generated_mask, u_score=None):
        self.defect_mask_list.append(generated_mask)
        # self.uncertainty_score_list.append(u_score)
    
    def input_img_score(self, score):
        self.img_score_list.append(score)
        
    
    def Find_Optimal_Cutoff(self, TPR, FPR, threshold):
        y = TPR - FPR
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        return optimal_threshold, point

    def segment(self, input, threshold=0.5):
        # binary score
        binary_scores = np.zeros_like(input)    # torch.zeros_like(scores)
        binary_scores[input < threshold] = 0
        binary_scores[input >= threshold] = 1
        return binary_scores

    def validate_single_image(self, img, label, rgb, idx):
        img_vec = img.ravel() / 255
        label_vec = label.ravel() / 255
        fpr, tpr, thresholds = roc_curve(label_vec.astype(np.uint8), img_vec)
        auroc = auc(fpr, tpr)
        rgb_ = rgb.copy()
        # optimal_th, _ = self.Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

        # get optimal threshold
        IoU = 0
        precision, recall, thresholds = precision_recall_curve(label_vec.astype(np.uint8), img_vec)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
        threshold = thresholds[np.argmax(f1)]

        binary_score = self.segment(img / 255, threshold)
        _, contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0,0,255), 2)
        # rgb[..., 1] = np.where(binary_score == 1, 255, rgb[..., 1])    #  uncommet for final results
        cv2.imwrite(os.path.join(self.img_save_path, f'result_{idx:03d}.png'), rgb)
        
        heat_map = np.float32(cv2.applyColorMap(img, cv2.COLORMAP_JET))
        heat_map = 0.6 * heat_map + 0.4 * np.float32(rgb_)
        heat_map = np.clip(heat_map, 0, 255)
        cv2.imwrite(os.path.join(self.img_save_path, f'heatmap_{idx:03d}.png'), np.uint8(heat_map))

        F1 = f1[np.argmax(f1)]
        precision = precision[np.argmax(f1)]
        recall = recall[np.argmax(f1)]

        return IoU, precision, recall, F1, fpr, tpr, auroc

    def calculate_single_defect(self, save_dir):

        log_save_path = save_dir
        IoU_mean = []
        Precise_mean = []
        Recall_mean = []
        F1_mean = []
        auroc_mean = []
        fpr_defect = {}
        tpr_defect = {}
        length = len(self.defect_mask_list)

        self.img_save_path = os.path.join(log_save_path, 'results')
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)

        #=== get images for pro_auc calculation
        self.scores.extend(self.defect_mask_list)
        self.masks.extend(self.gt_mask_list)

        print(length, len(self.gt_mask_list))
        if self.defect_name == 'good':
            self.gt_mask_list = np.repeat(self.gt_mask_list, length, axis=0)
        for idx, img in enumerate(self.defect_mask_list):
            label = self.gt_mask_list[idx]
            rgb = self.gt_img_list[idx]
            if self.defect_name == 'good':
                img = 255 - img
            IoU, precise, recall, F1, fpr, tpr, auroc = self.validate_single_image(img, label, rgb, idx)
            fpr_defect[idx] = fpr
            tpr_defect[idx] = tpr
            IoU_mean.append(IoU)
            Precise_mean.append(precise)
            Recall_mean.append(recall)
            F1_mean.append(F1)  
            auroc_mean.append(auroc)

        #calculate micro roc and auc
        all_fpr = np.unique(np.concatenate([fpr_defect[i] for i in range(length)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(length):
            mean_tpr += interp(all_fpr, fpr_defect[i], tpr_defect[i])
        mean_tpr /= length
        auroc_macro = auc(all_fpr, mean_tpr)

        IoU_mean = np.mean(IoU_mean)
        Precise_mean = np.mean(Precise_mean)
        Recall_mean = np.mean(Recall_mean)
        F1_mean = np.mean(F1_mean)     
        auroc_mean = np.mean(auroc_mean)

        self.AuROC[self.defect_name] = {'macro': auroc_macro, 'micro': auroc_mean}
        self.IoU[self.defect_name] = IoU_mean
        self.Precise[self.defect_name] = Precise_mean
        self.Recall[self.defect_name] = Recall_mean
        self.F1[self.defect_name] = F1_mean
        self.FPR[self.defect_name] = all_fpr
        self.TPR[self.defect_name] = mean_tpr
        # self.defect_mask_list = []    # clear mask list

        print(f'Defect: {self.defect_name}  \nAuROC: [macro]: {auroc_macro:.4f}, [micro]: {auroc_mean:.4f}')
        print('Mean IoU : {:.4f} Mean Precise : {:.4f}  Mean Recall : {:.4f}  Mean F1 : {:.4f}'.format(IoU_mean, Precise_mean, Recall_mean, F1_mean))
        print(f'==================== {self.model_name}: {self.defect_name}  ==================== ')
        filename = f'{self.defect_name}_result.txt'
        log_save_path = os.path.join(log_save_path, filename)
        t = time.localtime()
        with open(log_save_path, 'a') as f:
            f.write(f'===== [{t.tm_year}.{t.tm_mon}.{t.tm_mday} // {t.tm_hour} : {t.tm_min}] ===========\n')
            f.write(f'Defect: {self.defect_name}  \nAuROC: [macro]: {auroc_macro:.4f}, [micro]: {auroc_mean:.4f}\n')
            f.write('Mean IoU : {:.4f} Mean Precise : {:.4f}  Mean Recall : {:.4f}  Mean F1 : {:.4f} \n'.format(IoU_mean, Precise_mean, Recall_mean, F1_mean))
    
    def calculate_result(self, result_dir, time_cost):
        AuROC_ma = []
        AuROC_mi = []
        IoU = []
        Precise = []
        Recall = []
        F1_mean = []

        #image level auroc
        label = np.array(self.img_label_list).astype(np.uint8)
        score = np.array(self.img_score_list)
        fpr, tpr, _ = roc_curve(label, score)
        img_level_auroc = auc(fpr, tpr)

        self.img_score_list = []
        self.img_label_list = []

        pro_auc_score = per_region_overlap(self.scores, self.masks)

        filename = f'{self.model_name}_result.txt'
        log_save_path = os.path.join(result_dir, filename)
        t = time.localtime()
        with open(log_save_path, 'a') as f:
            f.write(f'=========== [{t.tm_year}.{t.tm_mon}.{t.tm_mday} // {t.tm_hour} : {t.tm_min}] ===========\n')
            for defect_name in self.F1:
                auroc = self.AuROC[defect_name]
                macro = auroc['macro']
                micro = auroc['micro']
                AuROC_ma.append(auroc['macro'])
                AuROC_mi.append(auroc['micro'])
                Precise.append(self.Precise[defect_name])
                Recall.append(self.Recall[defect_name])
                IoU.append(self.IoU[defect_name])
                F1_mean.append(self.F1[defect_name])
                
                f.write(f'Model: {self.model_name} ,  Defect: {defect_name}  \n')
                f.write(f'AuROC: [macro]: {macro:.4f}, [micro]: {micro:.4f} \n')
                f.write(f'Mean IoU : {self.IoU[defect_name]:.4f} Mean Precise : {self.Precise[defect_name]:.4f}  Mean Recall : {self.Recall[defect_name]:.4f}  Mean F1 : {self.F1[defect_name]:.4f} \n')
                f.write('----------------------\n')
 
            AuROC_ma = np.mean(AuROC_ma)
            AuROC_mi = np.mean(AuROC_mi)
            IoU = np.mean(IoU)
            Precise = np.mean(Precise)
            Recall = np.mean(Recall)
            F1_mean = np.mean(F1_mean)

            f.write(f'AuROC_ma: {AuROC_ma:.4f}, AuROC_mi: {AuROC_mi:.4f}\n')
            f.write(f'PRO_AUC 30% : {pro_auc_score:.4f}\n')
            f.write(f'Image_level_AUROC : {img_level_auroc:.4f}\n')
            f.write(f'IoU: {IoU:.4f}, Precise: {Precise:.4f}\n')
            f.write(f'Recall: {Recall:.4f}, F1_mean: {F1_mean:.4f}\n')
            f.write(f'Avg Process Time: {time_cost:.4} ms\n\n')
        
        # clear one model images
        self.scores = []
        self.masks = []

        result = collections.OrderedDict()
        result['AuROC_ma'] = AuROC_ma
        result['AuROC_mi'] = AuROC_mi
        result['IoU'] = IoU
        result['Precise'] = Precise
        result['Recall'] = Recall
        result['F1'] = F1_mean
        result['Avg_time'] = time_cost
        result['PRO_AUC'] = pro_auc_score
        result['Img_AUROC'] = img_level_auroc
        return result
        




