import torch
import torch.utils.data as data
import os
from PIL import Image
import cv2
import glob
import numpy as np
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
from PIL import ImageEnhance
from tqdm import tqdm
# import util.util as util 
import util.pySQI as pySQI
import imp
util = imp.load_source('util', './util/util.py')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


#  DataLoader ========================================

class CustomDatasetDataLoader():
    def __init__(self):
        super(CustomDatasetDataLoader, self).__init__()

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=opt.isTrain,
            drop_last=opt.isTrain,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
    
    def load_dataset(self):
        return self.dataset

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader


#  Dataset ============================================

def CreateDataset(opt):
    # dataset = mvtec_Dataset()
    if opt.data_format == 'Yolo':
        dataset = YoloDataset(opt)
    elif opt.data_format == 'test':
        dataset = test_Dataset(opt)
    print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset

def make_dataset(dir):
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images



#  mvtec dataset  =====================================

class mvtec_Dataset(data.Dataset):
    def __init__(self):
        super(mvtec_Dataset, self).__init__()

    def name(self):
        return 'mvtec_Dataset'
    
    def addsalt_pepper(self, img, SNR):
        img_ = img.copy()
        h, w, c = img_.shape
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        mask = np.repeat(mask, c, axis=2)     
        img_[mask == 1] = 255   
        img_[mask == 2] = 0     
        return img_

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        print(self.root)
        self.dir = os.path.join(opt.dataroot, opt.phase)  
        self.paths = sorted(make_dataset(self.dir))

        print('Preparing Dataset ...')
        self.input_list = []
        self.label_list = []

        for path in tqdm(self.paths):
            rgb = cv2.imread(path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # SNR = 0.9
            # noise_rgb = self.addsalt_pepper(rgb, SNR)

            self.input_list.append(rgb)
            self.label_list.append(rgb)
        
        self.dataset_size = len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        A = self.input_list[index]
        A = Image.fromarray(A)
        A_tensor = transform(A)

        B = self.label_list[index]
        B = Image.fromarray(B)
        B_tensor = transform(B)
        
        input_dict = {'input': A_tensor, 'label': B_tensor, 'path': path}
        return input_dict

    def __len__(self):
        return len(self.paths) // self.opt.batchSize * self.opt.batchSize

#  custom dataset =====================================

class YoloDataset(data.Dataset):
    def __init__(self, opt, image_size=(384,512), mosaic=True):
        super(YoloDataset, self).__init__()
        annotation_path = opt.data_annotation_path
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(1)
        np.random.shuffle(lines)
        np.random.seed(None)

        self.opt = opt
        self.train_lines = lines
        self.image_size = image_size
        self.mosaic = mosaic
        self.flag = True

        self.images, self.boxes, self.ori_imgs = self.data_process()

    def name(self):
        return 'yolo_dataset'

    def __len__(self):
        return len(self.images) // self.opt.batchSize * self.opt.batchSize

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def map_generation(self, img, boxes):
        h, w, _ = img.shape
        row = np.arange(h)
        row = row.reshape(h,1).repeat(w, axis=1)
        col = np.arange(w)
        col = col.reshape(1,w).repeat(h, axis=0)
        mask = np.zeros((h,w), dtype=np.uint8)

        c_points = []
        v_maps = []
        for box in boxes:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            c_point = [(y_min + y_max)//2, (x_min + x_max)//2]
            c_points.append(c_point)
            mask[y_min:y_max, x_min:x_max] = 1
            
            row_temp = row.copy().astype(np.float32).reshape(1,h,w)
            col_temp = col.copy().astype(np.float32).reshape(1,h,w)
            row_temp -= c_point[0]
            col_temp -= c_point[1]
            row_temp /= h
            col_temp /= w
            v_map = np.concatenate([row_temp, col_temp], axis=0)
            v_maps.append(v_map)
        v_maps = np.concatenate(v_maps, axis=0)
        mask = mask.astype(np.float32).reshape(1,h,w)
        c_points = np.array([c_points]).astype(np.float32).reshape(1, len(boxes), 2)
        return [mask, v_maps, c_points]

    def data_process(self, k_num = 11):
        images = []
        boxes_list = []
        ori_imgs = []
        for line in tqdm(self.train_lines):
            line = line.split()
            boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            if len(boxes) != k_num:
                continue
            img = cv2.imread(line[0])
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 384))
            ori_imgs.append(img)
            img = pySQI.SQI(img)
            images.append(img)
            boxes_list.append(boxes)
        return images, boxes_list, ori_imgs

    def __getitem__(self, index):
        img = self.images[index]
        boxes = self.boxes[index]
        ori_img = self.ori_imgs[index]

        y = np.array(sorted(boxes, key=lambda x: x[0]))
        y = np.array(sorted(y, key=lambda x: x[-1]))
        mask, v_maps, c_points = self.map_generation(img, y)

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)
        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_inp = torch.from_numpy(tmp_inp)
        mask = torch.from_numpy(mask)
        v_maps = torch.from_numpy(v_maps)
        c_points = torch.from_numpy(c_points)
        
        # tmp_targets = np.array(y, dtype=np.float32)

        input_dict = {'input': tmp_inp, 'mask':mask, 'vmap': v_maps, 'cpoints':c_points,'ori_img':ori_img}
        return input_dict

class test_Dataset(data.Dataset):
    def __init__(self, opt):
        super(test_Dataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot 
        self.gt_root = './dataset/gt'
        self.dir = os.path.join(opt.dataroot, opt.phase)  
        self.paths = sorted(make_dataset(self.dir))

        print('Preparing Dataset ...')
        self.input_list = []
        self.label_list = []
        self.data = []
        self.ori_data = []
        self.ori_imgs = []
        self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

        for path in tqdm(self.paths):
            rgb = cv2.imread(path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (512, 384))
            self.ori_imgs.append(rgb)
            rgb = pySQI.SQI(rgb)
            fn = path.split('/')[-1].split('.')[0] + '.png'
            gt_name = os.path.join(self.gt_root, fn)
            mask = cv2.imread(gt_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (512, 384))
            self.input_list.append(rgb)
            self.label_list.append(mask)
    
            # spade
            # img = cv2.imread(path)
            # self.ori_imgs.append(img)
            # m, n, _ = img.shape
            # img = cv2.resize(img, (n//8, m//8))
            # self.data.append(img)
            # self.ori_data.append(img)

        self.dataset_size = len(self.paths)

    def name(self):
        return 'test_Dataset'

    def map_generation(self, img, boxes):
        h, w, _ = img.shape
        row = np.arange(h)
        row = row.reshape(h,1).repeat(w, axis=1)
        col = np.arange(w)
        col = col.reshape(1,w).repeat(h, axis=0)
        mask = np.zeros((h,w), dtype=np.uint8)

        c_points = []
        v_maps = []
        for box in boxes:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            c_point = [(y_min + y_max)//2, (x_min + x_max)//2]
            c_points.append(c_point)
        #     mask[y_min:y_max, x_min:x_max] = 1
            
        #     row_temp = row.copy().astype(np.float32).reshape(1,h,w)
        #     col_temp = col.copy().astype(np.float32).reshape(1,h,w)
        #     row_temp -= c_point[0]
        #     col_temp -= c_point[1]
        #     row_temp /= h
        #     col_temp /= w
        #     v_map = np.concatenate([row_temp, col_temp], axis=0)
        #     v_maps.append(v_map)
        # v_maps = np.concatenate(v_maps, axis=0)
        # mask = mask.astype(np.float32).reshape(1,h,w)
        c_points = np.array([c_points]).astype(np.float32).reshape(1, len(boxes), 2)
        # return [mask, v_maps, c_points]
        return c_points

    def __getitem__(self, index):
        path = self.paths[index]
        
        A = self.input_list[index]
        # c_points = self.map_generation(A, y)
        A = np.array(A, dtype=np.float32)
        A = np.transpose(A / 255.0, (2, 0, 1))
        A_tensor = torch.from_numpy(A)

        B = self.label_list[index]
        B = np.array(B, dtype=np.float32)
        B = np.expand_dims(B / 255.0, axis = 0)
        B_tensor = torch.from_numpy(B)
        ori_img = self.ori_imgs[index]
        

        # spade part
        # img = self.data[index]
        # Img = Image.fromarray(img)
        # Img = self.transform(Img)
        # ori = self.ori_data[index]
        # Ori = Image.fromarray(ori)
        # Ori = self.transform(Ori)
        
        # input_dict = {'input': A_tensor, 'mask': B_tensor, 'path': path, 'Img':Img, 'Ori':Ori,'ori_img':ori_img}
        input_dict = {'input': A_tensor, 'mask': B_tensor, 'path': path,'ori_img':ori_img}

        return input_dict

    def __len__(self):
        return len(self.paths) // self.opt.batchSize * self.opt.batchSize



if __name__ == "__main__":
    dataset = YoloDataset()
    print(len(dataset))
    for i in range(len(dataset)):
        input_dict = dataset.__getitem__(i)
        print(i, input_dict['mask'].shape, input_dict['vmap'].shape, input_dict['cpoints'].shape)


