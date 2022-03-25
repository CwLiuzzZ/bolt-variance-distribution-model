from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from math import exp
from skimage.io import imread, imsave, imshow
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import imageio
from collections import OrderedDict
import util.global_variables as gl
from einops import rearrange, reduce
import tensorflow as tf

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
# import pytorch_ssim

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array

def tensor2heatmap(image_tensor, rgb_numpy, imtype=np.uint8, normalize=True):

    if image_tensor.is_cuda:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor.float().numpy()

    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  
    image_numpy = np.clip(image_numpy, 0, 1)
    image_numpy -= np.min(image_numpy)
    if np.max(image_numpy) != 0:
        image_numpy /= (np.max(image_numpy)- np.min(image_numpy))
    image_numpy = np.float32(cv2.applyColorMap(np.uint8(255*image_numpy), cv2.COLORMAP_JET))
    # cam = np.float32(rgb_numpy) + image_numpy
    cam_ = np.float32(rgb_numpy) / 255
    # cam = cam - np.min(cam)
    
    image_numpy /= 255
    cam = 0.5 * (image_numpy + cam_)
    image_numpy = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)

    return image_numpy
    
def heatmap2mask(image_tensor, nmask_tensor):
    image_numpy = image_tensor.cpu().float().numpy() * -1
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0
    if nmask_tensor is not None:
        nmask_numpy = nmask_tensor.float().numpy()
        nmask_numpy = (np.transpose(nmask_numpy, (1, 2, 0)) + 1) / 2.0
    else:
        nmask_numpy = np.ones(image_numpy.shape, np.uint8)
        
    image_numpy = np.clip(image_numpy, 0, 1)
    # image_numpy -= np.min(image_numpy)
    image_numpy *= nmask_numpy
    # if np.max(image_numpy) != 0:
    #     image_numpy /= (np.max(image_numpy)- np.min(image_numpy))
    image_numpy *= 255
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(np.uint8)

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    if image_tensor.is_cuda:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor.float().numpy()

    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) * 255 / 2.0 
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255
 
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]

    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_multiple_image(image_path,image_numpy1=None, image_numpy2=None,image_numpy3=None,image_numpy4=None):
    image_numpy = image_numpy1
    if not image_numpy2 is None:
        image_numpy=np.hstack((image_numpy,(np.ones((384,10))*255)))
        image_numpy=np.hstack((image_numpy,image_numpy2))
        if not image_numpy3 is None:
            image_numpy=np.hstack((image_numpy,(np.ones((384,10))*255)))
            image_numpy=np.hstack((image_numpy,image_numpy3))
            if not image_numpy4 is None:
                image_numpy=np.hstack((image_numpy,(np.ones((384,10))*255)))
                image_numpy=np.hstack((image_numpy,image_numpy4))
    image_pil = Image.fromarray(image_numpy)
    if image_pil.mode == "F":
        image_pil = image_pil.convert('RGB') 
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def rgb2contour(rgb_tensor):
    if rgb_tensor.is_cuda:
        rgb_tensor = rgb_tensor.cpu()
    rgb = np.uint8(255 * np.transpose(rgb_tensor.squeeze(0).numpy(), (1,2,0)))
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.convertScaleAbs(cv2.Sobel(rgb_gray, cv2.CV_16S, 1, 0))
    sobely = cv2.convertScaleAbs(cv2.Sobel(rgb_gray, cv2.CV_16S, 0, 1))
    contour = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    contour = Image.fromarray(contour).convert('RGB')
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
    contour_tensor = transform(contour).unsqueeze(0)
    return contour_tensor

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_args(train=True, spade = False, students = False):
    path = 'config/'
    if train:
        print("load config from train_config.txt")
        filename = 'train_config.txt'
    elif spade:
        print("load config from spade_config.txt")
        filename = 'spade_config.txt'
    elif students:
        print("load config from students_config.txt")
        filename = 'students_config.txt'
    else:
        print("load config from test_config.txt")
        filename = 'test_config.txt'
    filename = path + filename

    with open(filename, 'r') as f:
        lines = f.readlines()
        opt = []
        for line in lines:
            sp = line.strip().split(' ')
            opt.extend(sp)
    return opt

def residual_map(generated_img, original_img, mode='ssim', bright = True):
    # all inputs are tensor
    # l2 is the mode flag
    if mode == 'l2':
        generated_img = generated_img.cpu().float().numpy()
        original_img = original_img.float().numpy()
        res_img = np.max(np.square(original_img - generated_img), axis=1)    
        res_img_norm = torch.from_numpy(res_img).float()
        res_img_norm = res_img_norm.unsqueeze(1)

    elif mode == 'l1':
        generated_img = generated_img.cpu().float().numpy()
        original_img = original_img.float().numpy()
        res_img = np.max(np.abs(original_img - generated_img), axis=1)    
        res_img_norm = torch.from_numpy(res_img).float()
        res_img_norm = res_img_norm.unsqueeze(1)

    elif mode == 'ssim':
        generated_img = generated_img
        original_img = original_img.cuda()
        res_img = DSSIM(generated_img, original_img, bright)   
        res_img = np.max(res_img, axis=1)
        res_img_norm = torch.from_numpy(res_img).float()
        res_img_norm = res_img_norm.unsqueeze(1)

    return res_img_norm

def Lcontour(contours):
    area = []
    for i in contours:
        area.append(cv2.contourArea(i))
    area = np.array(area)
    try:
        index = np.argmax(area)
    except:
        return None
    return index

def DSSIM(output, target, bright = True):
    _, ssim_map = ssim(output, target, bright=bright)
    residual = torch.ones(ssim_map.shape) - ssim_map.detach().cpu()
    res_map = residual.numpy()
    return res_map

# ================  image visulization =================================
def visulization(img_file, mask_path, score_map_path, saving_path):
    # image name
    img_name = img_file.split("/")
    img_name = "-".join(img_name[-2:])

    # image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    #     imsave("feature_maps/Results/gt_image/{}".format(img_name), image)

    # mask
    mask_file = os.path.join(mask_path, img_name)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # binary score {0, 255}
    score_file = os.path.join(score_map_path, img_name)
    score = cv2.imread(score_file, cv2.IMREAD_GRAYSCALE)
    img = img[:, :, ::-1]  # bgr to rgb
    img[..., 1] = np.where(score == 255, 255, img[..., 1])

    # save
    imsave(os.path.join(saving_path, "{}".format(img_name)), img)  

def visulization_score(img_file, mask_path, score_map_path, saving_path):
    # image name
    img_name = img_file.split("/")
    img_name = "-".join(img_name[-2:])

    # image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    #     imsave("feature_maps/Results/gt_image/{}".format(img_name), image)

    superimposed_img = img.copy()

    # mask
    mask_file = os.path.join(mask_path, img_name)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=-1)
    img = img[:, :, ::-1]  # bgr to rgb

    # normalized score {0, 255}
    score_file = os.path.join(score_map_path, img_name)
    score = cv2.imread(score_file, cv2.IMREAD_GRAYSCALE)

    heatmap = cv2.applyColorMap(score, cv2.COLORMAP_JET)  # 灏唖core杞�鎹㈡垚鐑�鍔涘浘
    superimposed_img = heatmap * 0.7 + superimposed_img * 0.8     # 灏嗙儹鍔涘浘鍙犲姞鍒板師鍥惧�?????
    # cv2.imwrite('cam.jpg', superimposed_img)  # 灏嗗浘鍍忎繚瀛�

    # save
    cv2.imwrite(os.path.join(saving_path, "{}".format(img_name)), superimposed_img)
    imsave(os.path.join(saving_path, "gt_{}".format(img_name)), img)

  
###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

# ====================================   SSIM 
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, bright = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01
    C2 = 0.03

    if bright:
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    else:
        ssim_map = (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)

    if size_average:
        return ssim_map.mean(), ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1), ssim_map

def ssim(img1, img2, window_size = 11, size_average = True, bright = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average, bright)

# ============================= BBOX ==================================

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        璁＄畻IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue
                
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue
                
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix

def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def visualize_loc_result(test_imgs, ori_imgs, score, score_map_list, threshold, save_path, count):
    length = len(test_imgs)
    for t_idx in range(length):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        ori_img = ori_imgs[t_idx]
        ori_img = denormalization(ori_img)

        test_pred = score_map_list[0]

        mask = np.zeros(test_pred.shape, dtype=np.uint8)
        mask[test_pred > threshold] = 255
        # test_pred_img = np.zeros(test_img.shape, dtype=np.uint8)
        # test_pred_img[mask == 255] = test_img[mask == 255] 

        fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.suptitle(f'Score: {np.max(test_pred):.3f}')

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        # roi = self.roi_extraction(test_img)
        # mask[roi<255] = 0

        _, contours, _, = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # c_max = []
        # for i in range(len(contours)):
        #     cnt = contours[i]
        #     c_max.append(cnt)
        
        result = cv2.drawContours(ori_img.copy(), contours, -1, (255,0,0), 2)
        result_ = cv2.drawContours(ori_img.copy(), contours, -1, (0,0,255), 2)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(mask, cmap='gray')
        ax_img[1].title.set_text('Predicted mask')
        ax_img[2].imshow(result)
        ax_img[2].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%03d.png' % (count)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)

        os.makedirs(os.path.join(save_path, 'results'), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, 'results', '%03d.png' % (count)), result_)


# #  鍙�瑙嗗寲鐗瑰緛鍥�
# def show_feature_map(feature_map):
#     feature_map = feature_map.squeeze(0)
#     feature_map = feature_map.cpu().numpy()
#     feature_map_num = feature_map.shape[0]
#     row_num = np.ceil(np.sqrt(feature_map_num))
#     plt.figure()
#     for index in range(feature_map_num):
#         plt.subplot(row_num, row_num, index+1)
#         plt.imshow(feature_map[index], cmap='gray')
#         plt.axis('off')
#         folder_path = './results/defect/feature_map'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         imageio.imwrite(os.path.join(folder_path,str(index+1)+".png"), feature_map[index])
#         # scipy.misc.imsave(str(index)+".png", feature_map[index-1])

def increment_mean_and_var(mu_N, var_N, N, batch):
    '''calculate mean and variance on channel
       Increment value of mean and variance based on
       current mean, var and new batch
    '''
    # batch: (batch, vector, h, w)
    B = batch.size()[0] # batch size
    # we want a descriptor vector -> mean over batch and pixels
    print(batch.size())
    if len(batch.size()) == 4:
        dim = [0,2,3]
    elif len(batch.size()) == 3:
        dim = [0,1,2]
    mu_B = torch.mean(batch, dim)
    S_B = B * torch.var(batch, dim, unbiased=False) 
    S_N = N * var_N
    mu_NB = N/(N + B) * mu_N + B/(N + B) * mu_B
    S_NB = S_N + S_B + B * mu_B**2 + N * mu_N**2 - (N + B) * mu_NB**2
    var_NB = S_NB / (N+B)
    return mu_NB, var_NB, N + B

def load_student_model(model, model_path):
    '''load students model dict
    '''
    model_name = model_path.split('/')[-1]
    try:
        print(f'Loading of {model_name} succesful.')
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print(e)
        print('No model available.')
        print(f'Initilialisation of random weights for {model_name}.')

def resize_teacher_feature(array,d1,d3,d4):
    ''' expand teacher's mean value and var value from a array 
        to match teachers's feature's size
        (v) -> (batch_size,v,h,w)
    '''
    list = []
    for item in array:
        arr = np.array([item])
        arr = arr.repeat(d3*d4)
        torch_data=torch.from_numpy(arr)
        torch_data = torch.reshape(torch_data,(1,1,d3,d4))
        list.append(torch_data)
    tensor = torch.cat(list,1)
    tensor = tensor.repeat(d1,1,1,1)
    tensor = tensor.cuda()
    return tensor

def get_hook_orderdict(outputs):
    dict = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
    for k, v in zip(dict.keys(), outputs):
        dict[k].append(v)
    for k, v in dict.items():
        dict[k] = torch.cat(v, 0)
    outputs.clear()
    return dict

def get_layer_ave_hook_orderdict(outputs):
    dict = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
    new_output = []
    len_ = int(len(outputs)/2)
    # get average for layer[-1] and layer[-2]
    for i in range(len_):
        new_output.append((outputs[2*i]+outputs[2*i+1])/2)
    for k, v in zip(dict.keys(), new_output):
        dict[k].append(v)
    for k, v in dict.items():
        dict[k] = torch.cat(v, 0)
    outputs.clear()
    return dict

def get_normed_orderdict(outputs,mu,var):
    dict = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
    new_output = []
    len_ = int(len(outputs)/2)
    # get average for layer[-1] and layer[-2]
    for i in range(len_):
        new_output.append((outputs[2*i]+outputs[2*i+1])/2)
    for k, v in zip(dict.keys(), new_output):
        dict[k].append(v)
    for k, v in dict.items():
        dict[k] = torch.cat(v, 0)
        dict[k] = (dict[k] - mu[k]) / torch.sqrt(var[k])
    outputs.clear()
    return dict

# 灏唂eature_map涓婄殑鐗瑰緛鎸�?�収key_map涓婄殑鍏抽敭鐐�?�彁鍙栵紝鍙�鎻�?????彇鍏抽敭鐐圭殑鐗瑰緛
def feat_extraction(kp_map, feature_map, smaller=True):
    dim_kp = kp_map.dim()
    dim_feat = feature_map.dim()
    x,y,m,n = feature_map.shape
    kp_map[np.where(kp_map > 0)] = 1
    if smaller:
        if dim_kp == 4:
            kp_map = torch.squeeze(kp_map)
            kp_map = F.interpolate(kp_map.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(y,m,n),
                        mode='trilinear', align_corners=False)
            kp_map = kp_map.squeeze(0)
            # print(kp_map.shape)
        elif dim_kp == 2:
            kp_map = F.interpolate(kp_map.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(y,m,n),
                        mode='trilinear', align_corners=False)
            kp_map = kp_map.squeeze(0)
            # print(kp_map.shape)
        else:
            print('dim error')
    else:
        if dim_kp == 4:
            a,b,c,d = kp_map.shape
            feature_map = torch.squeeze(feature_map)
            # feature_map = F.interpolate(feature_map, size=(c,d),
            #                 mode='bilinear', align_corners=False)
            feature_map = F.interpolate(feature_map.unsqueeze(0).unsqueeze(0), size=(b,c,d),
                            mode='trilinear', align_corners=False)
            feature_map = feature_map.squeeze(0)
            # print(feature_map.shape)
        elif dim_kp == 2:
            kp_map = kp_map.unsqueeze(0).unsqueeze(0)
            a,b,c,d = kp_map.shape
            feature_map = torch.squeeze(feature_map)
            feature_map = F.interpolate(feature_map.unsqueeze(0).unsqueeze(0), size=(b,c,d),
                            mode='trilinear', align_corners=False)
            # feature_map = F.interpolate(feature_map, size=(c,d),
            #                 mode='bilinear', align_corners=False)
            feature_map = feature_map.squeeze(0)
            # print('feature_map.shape',feature_map.shape)
    kp_map=kp_map.to(device='cuda')
    feat_keypoint = kp_map * feature_map

    return kp_map, feature_map, feat_keypoint

# feature visualise
def visualize_key_result(test_imgs, ori_imgs, feat_keypoint, threshold, save_path, count):
    feat_keypoint = F.interpolate(feat_keypoint, size=(193,258),
                            mode='bilinear', align_corners=False)
    feat_keypoint = feat_keypoint.squeeze(0)

    length = len(test_imgs)
    for t_idx in range(length):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        ori_img = ori_imgs[t_idx]
        ori_img = denormalization(ori_img)

        test_pred = feat_keypoint[0]

        mask = np.zeros(test_pred.shape, dtype=np.uint8)
        mask[test_pred > 0 ] = 255
        # test_pred_img = np.zeros(test_img.shape, dtype=np.uint8)
        # test_pred_img[mask == 255] = test_img[mask == 255] 

        fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # plt.suptitle(f'Score: {np.max(test_pred):.3f}')

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        # roi = self.roi_extraction(test_img)
        # mask[roi<255] = 0

        # print(type(mask))
        # print(mask.shape)
        # print(type(mask[0,0,0]))
        _, contours, _, = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # c_max = []
        # for i in range(len(contours)):
        #     cnt = contours[i]
        #     c_max.append(cnt)
        
        result = cv2.drawContours(ori_img.copy(), contours, -1, (255,0,0), 2)
        result_ = cv2.drawContours(ori_img.copy(), contours, -1, (0,0,255), 2)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(mask, cmap='gray')
        ax_img[1].title.set_text('Predicted mask')
        ax_img[2].imshow(result)
        ax_img[2].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%03d.png' % (count)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)

        os.makedirs(os.path.join(save_path, 'results'), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, 'results', '%03d.png' % (count)), result_)
 
def draw_CAM_util(feature,img,append_name, save_count,key,transform=None, visual_heatmap=False):
    grads = feature
    # pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # pooled_grads = pooled_grads[0]
    feature = feature[0]
    # feature
    for i in range(feature.shape[0]):
        # feature[i, ...] *= pooled_grads[i, ...]

        # transform heatmap
        heatmap = feature.cpu().detach().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # show pic
        if visual_heatmap:
            plt.matshow(heatmap)
            plt.show()

        # img = cv2.imread(img_path) 
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # resize heat map to img's size
        heatmap = np.uint8(255 * heatmap)  
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
        superimposed_img = heatmap * 0.4 + img  # heat map dense
        save_path = './results/CAM/{}_{}_img{}.jpg'.format(append_name,key,save_count)
        cv2.imwrite(save_path, superimposed_img)  # save img

def draw_CAM(features, data,append_name, save_count,keys=None, transform=None, visual_heatmap=False):
    '''
     Class Activation Map
    :param features: input feature: single or dic of tensor in 4 dimension
    :param data: should contain ori_img
    :param save_path: CAM save path
    :param transform: 
    :param visual_heatmap: print pic
    :return:
    '''
    
    img = data['ori_img'][0].numpy()
    # multiple features
    if not keys is None:
        for key in keys: 
            feature = features[key].clone()
            draw_CAM_util(feature,img,append_name, save_count,key,transform, visual_heatmap)
    else:
        feature = features.clone()
        draw_CAM_util(feature,img,append_name, save_count,'main',transform, visual_heatmap)

def interpolation_train_layers(train_outputs):
    # interpolate layer1,2,3 to match layer4's size
    new_train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
    new_train_outputs_layer1_list = []
    new_train_outputs_layer2_list = []
    new_train_outputs_layer3_list = []
    for j in range(train_outputs['layer1'].shape[0]):
        new_train_outputs['layer1'] = train_outputs['layer1'][j,:,:,:]
        new_train_outputs['layer1'] = F.interpolate(torch.squeeze(new_train_outputs['layer1']).unsqueeze(0).unsqueeze(0), size=(512,48,64), 
                mode='trilinear', align_corners=False)
        new_train_outputs['layer1'] = new_train_outputs['layer1'].squeeze(0)
        new_train_outputs_layer1_list.append(new_train_outputs['layer1'])
        new_train_outputs['layer1'] = torch.cat(new_train_outputs_layer1_list, 0)

    for j in range(train_outputs['layer2'].shape[0]):
        new_train_outputs['layer2'] = train_outputs['layer2'][j,:,:,:]
        new_train_outputs['layer2'] = F.interpolate(torch.squeeze(new_train_outputs['layer2']).unsqueeze(0).unsqueeze(0), size=(512,48,64), 
                mode='trilinear', align_corners=False)
        new_train_outputs['layer2'] = new_train_outputs['layer2'].squeeze(0)
        new_train_outputs_layer2_list.append(new_train_outputs['layer2'])
        new_train_outputs['layer2'] = torch.cat(new_train_outputs_layer2_list, 0)
    
    for j in range(train_outputs['layer3'].shape[0]):
        new_train_outputs['layer3'] = train_outputs['layer3'][j,:,:,:]
        new_train_outputs['layer3'] = F.interpolate(torch.squeeze(new_train_outputs['layer3']).unsqueeze(0).unsqueeze(0), size=(512,48,64), 
                mode='trilinear', align_corners=False)
        new_train_outputs['layer3'] = new_train_outputs['layer3'].squeeze(0)
        new_train_outputs_layer3_list.append(new_train_outputs['layer3'])
        new_train_outputs['layer3'] = torch.cat(new_train_outputs_layer3_list, 0)

    new_train_outputs['layer4'] = train_outputs['layer4']
    return new_train_outputs


def interpolation_test_layers(test_outputs):
    new_test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
    # ori_test_outputs = test_outputs[:]
    new_test_outputs['layer1'] = F.interpolate(torch.squeeze(test_outputs['layer1']).unsqueeze(0).unsqueeze(0), size=(512,48,64),
                mode='trilinear', align_corners=False)
    new_test_outputs['layer1'] = new_test_outputs['layer1'].squeeze(0)

    new_test_outputs['layer2'] = F.interpolate(torch.squeeze(test_outputs['layer2']).unsqueeze(0).unsqueeze(0), size=(512,48,64),
                mode='trilinear', align_corners=False)
    new_test_outputs['layer2'] = new_test_outputs['layer2'].squeeze(0)

    new_test_outputs['layer3'] = F.interpolate(torch.squeeze(test_outputs['layer3']).unsqueeze(0).unsqueeze(0), size=(512,48,64),
                mode='trilinear', align_corners=False)
    new_test_outputs['layer3'] = new_test_outputs['layer3'].squeeze(0)
    new_test_outputs['layer4'] = test_outputs['layer4']
    return new_test_outputs

def keypoint_feature_map(keypoint_maps,feature_map):
    keys = ['layer1','layer2','layer3','layer4']
    keypoint_maps = torch.Tensor(tensor2im(keypoint_maps.data[0], normalize=False))
    keypoint_feature_map = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
    for key in keys:
        _, _, keypoint_feature_map[key] = feat_extraction(keypoint_maps, feature_map[key], smaller=True)
        keypoint_feature_map[key].unsqueeze(0)
        # keypoint_feature_map[key] = F.interpolate(keypoint_feature_map[key].unsqueeze(0), size=(512,48,64), 
        #             mode='trilinear', align_corners=False)
        # keypoint_feature_map[key] = keypoint_feature_map[key].squeeze(0)
    return keypoint_feature_map

def draw_array_png(arr1,arr2,path,x_name='x',y_name='y',title='title'):
    x = np.linspace(0, len(arr1), len(arr1))
    plt.plot(x,arr1,c = 'b',lw=0.5)
    if len(arr2) == len(arr1):
        plt.plot(x,arr2,c = 'y',lw=0.5)
    plt.axis('tight')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.savefig(path)
    plt.close()
    
def get_variance_map(students_pred):
    # student: (batch, student_id, h, w, vector)
    sse = reduce(students_pred**2, 'id vec h w -> id vec h w', 'sum')
    msse = reduce(sse, 'id vec h w -> vec h w', 'mean')
    mu_students = reduce(students_pred, 'id vec h w -> vec h w', 'mean')
    var = msse - reduce(mu_students**2, ' vec h w -> vec h w', 'sum')
    return var.unsqueeze(0)

def get_mean_var(X,dim):  
    assert len(X.shape) in (2, 4)        
    if len(X.shape) == 2:            
        mean = X.mean(dim)            
        var = ((X-mean)**2).mean(dim)        
    else: # X.shape = [B,C,H,W]            
        mean = X.mean(dim, keepdims=True)   
        var = ((X-mean)**2).mean(dim, keepdims=True)   
    return mean,var

def residual_map_process(residual_map,h,w,keypoint):
    '''
    output:
        residual_map: keypoint residual_map
        global_residual_map: residual_map of whole map
    '''
    residual_map = torch.mean(residual_map, dim=1, keepdim=True)
    # residual_map = torch.nn.functional.interpolate(residual_map, size=(h,w), mode="bilinear", align_corners=False)
    global_residual_map = residual_map.clone()
    residual_map[keypoint == 0] = 0
    # max = torch.max(residual_map)
    # min = torch.min(residual_map)
    # residual_map = (residual_map-min)/(max-min)
    # global_residual_map = (global_residual_map-min)/(max-min)
    # global_residual_map = torch.clamp(global_residual_map, 0, 1)
    return residual_map,global_residual_map

def get_kp_vertex(each_kp_map,vertex):
    vertex_ = vertex.clone()
    for i in range(each_kp_map.shape[0]):
        vertex_[0][2*i][each_kp_map[i]==1] = 0;
        vertex_[0][2*i+1][each_kp_map[i]==1] = 0;
    return vertex_

'''

    # 屏蔽所有层�?
    for c in range(22):
        for k in range(11):
            if not k == 9:
                vertex_[0][c][each_kp_map[k]>0] = 0;
    
    for i in range(each_kp_map.shape[0]):
        
        if not i == 10:
            vertex_[0][2*i][each_kp_map[1]>=0] = 0;
            vertex_[0][2*i+1][each_kp_map[1]>=0] = 0;
        else:
            continue
    右下
    0.0003
    28 16 26 27 20 18 15 15 18 34 31 mid=20
    623, 46, 604, 667, 328, 278, 173, 165, 0, 836, 700
    
    左上异常
    0.00007
        48 52 38 43 31 35 32 26 62 47 48 mid = 43
        0, 338, 53, 3, 125, 216, 274, 409, 623, 59, 85
    
    右下异常
    0.0002
    34 30 35 38 34 30 34 63 27 44 40 mid = 34
    338, 0, 362, 377, 180, 174, 107, 145, 46, 533, 448
    屏蔽1-100 550-inf
    #2?
    #0 + near[#3 ~ #4]
'''

def kp_circle_on_image(data, kp_coordinate,name):
    _, num_kp, _ = kp_coordinate.shape
    kp_on_image = data['ori_img'].squeeze().cpu().detach().numpy().copy()
    for i in range(num_kp):
        x, _, y, _ = kp_coordinate[0, i, :]
        kp_on_image = cv2.circle(kp_on_image, (x,y), 3, (255,0,0), -1)
    cv2.imwrite('./results/keypoint_map/'+name+'.png', kp_on_image)
    return kp_on_image

def kp_circle_on_image_lwe(data,name, kp1,kp2=None):
    kp_on_image = data['ori_img'].squeeze().cpu().detach().numpy().copy()
    for i in range(kp1.shape[0]):
        [y1,x1] = kp1[i]
        kp_on_image = cv2.circle(kp_on_image, (x1,y1), 3, (255,0,0), -1)
        if not kp2 is None:
            [y2,x2] = kp2[i]
            kp_on_image = cv2.circle(kp_on_image, (x2,y2), 3, (0,255,0), -1)
    cv2.imwrite('./results/keypoint_map/'+name+'.png', kp_on_image)
    return kp_on_image

def get_vertex_loss(vertex,vertex_ave,vertex_1,vertex_2,vertex_3,h,w,keypoint,each_kp_map,loss_fn):
    # vertex = get_kp_vertex(each_kp_map,vertex)
    # vertex_ave = get_kp_vertex(each_kp_map,vertex_ave)
    # vertex_1 = get_kp_vertex(each_kp_map,vertex_1)
    # vertex_2 = get_kp_vertex(each_kp_map,vertex_2)
    # vertex_3 = get_kp_vertex(each_kp_map,vertex_3)

    # get t_residual_map
    t_residual_map = loss_fn(vertex,vertex_ave)
    t_residual_map,t_global_residual_map = residual_map_process(t_residual_map,h,w,keypoint)

    # get s_residual_map
    # batchsize must = 1
    vertex_ = torch.vstack((vertex_1,vertex_2,vertex_3))
    s_residual_map = get_variance_map(vertex_)
    # _,s_residual_map = get_mean_var(vertex_,0)
    # # 屏蔽某区域�?�的其他所有�?�测
    # for i in range(22):
    #     s_residual_map[0][i][each_kp_map[1]==0] = 0;
    s_residual_map = torch.mean(s_residual_map, dim=1, keepdim=True)
    s_residual_map,s_global_residual_map = residual_map_process(s_residual_map,h,w,keypoint)
    
    return t_residual_map,t_global_residual_map,s_residual_map,s_global_residual_map

def draw_circle(input_img,keypoint_list,color=(255,0,0)):
    num_kp = len(keypoint_list)
    _,  h, w, _ = input_img.shape
    input_img = (input_img[0]/255.0).cpu().detach().numpy()
    for i in range(num_kp):
        y, x = keypoint_list[i]
        # cv2.rectangle(temp, (int(x)-18,int(y)-18), (int(x)+18,int(y)+18),(10,10,10), -1)
        
        # required [1, 3, 384, 512]
        cv2.ellipse(input_img, (int(x),int(y)), (15,15), 0, 0, 360, color, 2)
    circleed_img = torch.from_numpy(input_img).unsqueeze(0)
    circleed_img = rearrange(circleed_img,'b h w c -> b c h w')
    return circleed_img

def tensor_bound(label):
    one = tf.ones_like(label)

    zero = tf.zeros_like(label)

    label = tf.where(label <5, x=zero, y=one)
    return label