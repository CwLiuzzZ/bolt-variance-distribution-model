import numpy as np
import torch
import math
import glob
import time
import os
import ntpath
from collections import OrderedDict
import util.global_variables as gl
from torch.autograd import Variable
import torch.nn.functional as F
from options.train_options import TrainOptions
from options.test_options import TestOptions
from options.students_options import StudentsOptions
from options.spade_options import SpadeOptions
from data.data_loader import CreateDataLoader
from models.create_model import create_model 
import util.util as util
from util.visualizer import Visualizer
from torchvision.utils import make_grid
from train import train_pvmodel, train_student_net,valid_student,cal_kpZones_loss_gauss_distr,cal_mask_gauss_distr
from evaluation import evaluation
import shutil
from models import networks
import pickle
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

# def z1(x,y):
#     return 0.02 * x + 0.015 * y - 0.00008 * x * y + 0.00007 * x * x - 0.00002 * y * y + 15
# x = np.linspace(0,100,40,endpoint = False) # (40)
# y = np.linspace(0,300,40,endpoint = False)
# X, Y = np.meshgrid(x, y)
# Z1 = z1(X,Y) # (40,40)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# # ax.plot_wireframe(X, Y, Z1,color = 'orange')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('variance')
# # ax.view_init(60, 35)
# ax.set_title('surface')
# plt.savefig('./checkpoints/defect/3d.jpg')
# exit()

# from models.students_model import AnomalyNet
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

class Train_and_Valid():
    def __init__(self):
        self.result_path = './results'
        self.result_dict = OrderedDict()
        self.name = 'defect'
        self.count = 1
        # threshold: for spm
        self.threshold = 7

    def train(self):
        self.opt = TrainOptions().parse()
        self.visualizer = Visualizer(self.opt)
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        datasize = len(data_loader)
        train_pvmodel(self.opt, dataset, self.visualizer, datasize)

    def train_students(self):
        print("start load opts")
        print("---------------")
        try:
            del self.opt
        except Exception as e:
            print("no existed opt")

        self.opt = TrainOptions().parse(train = True, students = False)
        print('batchsize',self.opt.batchSize)
        self.visualizer = Visualizer(self.opt)
        print("complete load opts\n")
        
        print("start load teacher model")
        print("---------------")
        self.opt.model = 'pvnet'
        teacher = create_model(self.opt)
        print("complete load teacher model\n")

        self.opt.continue_train=False
        print("start create student model")
        print("---------------")
        self.opt.model = 'student'
        students = []
        for stu_id in range(self.opt.n_students):
            self.opt.stu_id = stu_id+1
            students.append(create_model(self.opt))
        print("complete create student model\n")

        print("start load dataset")
        print("---------------")
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        datasize = len(data_loader)
        self.opt.data_annotation_path = self.opt.valid_data_annotation_path
        valid_data_loader = CreateDataLoader(self.opt)
        valid_dataset = valid_data_loader.load_data()
        print("complete load dataset\n")

        print("start train student model")
        print("---------------")
        for index,student in enumerate(students):
            print('start train student {}/{}'.format(index+1,self.opt.n_students))
            train_student_net(self.opt, teacher, student, dataset, self.visualizer, datasize,valid_dataset)
        
        
    def train_students_gauss(self):
        print("start load opts")
        print("---------------")
        try:
            del self.opt
        except Exception as e:
            print("no existed opt")

        self.opt = TrainOptions().parse(train = True, students = False)
        self.opt.batchSize = 1
        self.visualizer = Visualizer(self.opt)
        print("complete load opts\n")
        
        print("start load teacher model")
        print("---------------")
        self.opt.model = 'pvnet'
        teacher = create_model(self.opt)
        print("complete load teacher model\n")

        print("start create student model")
        print("---------------")
        self.opt.model = 'student'
        students = []
        for stu_id in range(self.opt.n_students):
            self.opt.stu_id = stu_id+1
            students.append(create_model(self.opt))
        print("complete create student model\n")

        print("start load dataset")
        print("---------------")
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        print("complete load dataset\n")

        print("start train student kp zone loss gauss distribution")
        print("---------------")
        cal_kpZones_loss_gauss_distr(self.opt,teacher,students,dataset)
        
    def train_teacher_gauss(self):
        print("start load opts")
        print("---------------")
        try:
            del self.opt
        except Exception as e:
            print("no existed opt")

        self.opt = TrainOptions().parse(train = True, students = False)
        self.opt.batchSize = 1
        self.visualizer = Visualizer(self.opt)
        print("complete load opts\n")
        
        print("start load teacher model")
        print("---------------")
        self.opt.model = 'pvnet'
        teacher = create_model(self.opt)
        print("complete load teacher model\n")

        print("start create student model")
        print("---------------")
        self.opt.model = 'student'
        students = []
        for stu_id in range(self.opt.n_students):
            self.opt.stu_id = stu_id+1
            students.append(create_model(self.opt))
        print("complete create student model\n")

        print("start load dataset")
        print("---------------")
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        print("complete load dataset\n")

        print("start train teacher mask gauss distribution")
        print("---------------")
        cal_mask_gauss_distr(self.opt,teacher,dataset) 
        
    def test(self):
        print("start load opts")
        print("---------------")
        try:
            del self.opt
        except Exception as e:
            print("no existed opt")
        self.opt = TestOptions().parse(train=False, save=False, spade=False, students=False)
        self.opt.name = 'defect'
        self.opt.dataroot = os.path.join('./dataset', 'defect')
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.noise_repeat_num = 1   # test code only supports noise_repeat_num = 1
        self.opt.no_flip = True  # no flip
        print("complete load opts\n")
        
        print("start load model")
        print("---------------")
        self.opt.model = 'pvnet'
        pvnet = create_model(self.opt)

        self.opt.model = 'student'
        students = []
        for stu_id in range(self.opt.n_students):
            self.opt.stu_id = stu_id+1
            students.append(create_model(self.opt))

        print("start load dataset")
        print("---------------")
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        print("complete load dataset\n")

        time_cost = []
        result_dir = os.path.join(self.result_path, self.name)
        save_dir = os.path.join(result_dir, 'images')
        util.mkdirs([result_dir, save_dir])
        print("complete load model\n")

        for i, data in enumerate(tqdm(dataset)):
            # data: dict_keys(['input', 'mask', 'path', 'Img', 'Ori'])
            start_time = time.time()
            output_mask, keypoint, dis_map, teacher_feats,_,each_kp_map,_ = pvnet.inference(data)

            residual_map_list = []
            for student in students:
                residual_map_list.append(student.inference(data, teacher_feats, keypoint))
            residual_map = sum(residual_map_list)/self.opt.n_students

            time_cost.append(time.time() - start_time)
            visuals = OrderedDict([
                                    ('input', util.tensor2im(data['input'][0], normalize=False)),
                                    ('input_mask', util.tensor2im(data['mask'][0], normalize=False)),
                                    # ('Img', util.tensor2im(data['Img'][0], normalize=False)),
                                    # ('Ori', util.tensor2im(data['Ori'][0], normalize=False)),
                                    ('generated_mask', util.tensor2im(output_mask.data[0], normalize=False)),
                                    # ('map', util.tensor2im(keypoint.data[0], normalize=False)),
                                    # ('dis_map', util.tensor2im(dis_map.data[0], normalize=False)),
                                    ('residual_map', util.tensor2im(residual_map.data[0], normalize=False)),
                ])
            
            image_path = data['path']
            short_path = ntpath.basename(image_path[0])
            name = os.path.splitext(short_path)[0]
            image_name = '%s_%s.jpg' % (name,'residual_compare')
            save_path = os.path.join(save_dir, image_name)
            util.save_multiple_image(save_path,visuals['generated_mask'],visuals['input_mask'],visuals['residual_map'])
            for label, image_numpy in visuals.items():
                if not label == 'generated_mask' and not label == 'input_mask':
                    image_name = '%s_%s.jpg' % (name, label)
                    save_path = os.path.join(save_dir, image_name)
                    util.save_image(image_numpy, save_path)

        time_cost = np.mean(time_cost) * 1000
        print(f'AVG time cost: {time_cost:4f}')

    def test_upbone(self,if_gauss=False):
        print("start load opts")
        print("---------------")
        try:
            del self.opt
        except Exception as e:
            print("no existed opt")
        self.opt = TestOptions().parse(train=False, save=False, spade=False, students=False)
        self.opt.name = 'defect'
        self.opt.dataroot = os.path.join('./dataset', 'defect')
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.noise_repeat_num = 1   # test code only supports noise_repeat_num = 1
        self.opt.no_flip = True  # no flip
        print("complete load opts\n")
        
        print("start load model")
        print("---------------")
        self.opt.model = 'pvnet'
        pvnet = create_model(self.opt)

        self.opt.model = 'student'
        students = []
        for stu_id in range(self.opt.n_students):
            self.opt.stu_id = stu_id+1
            students.append(create_model(self.opt))
        print("complete load model\n")

        print("start load dataset")
        print("---------------")
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        print("complete load dataset\n")

        time_cost = []
        result_dir = os.path.join(self.result_path, self.name)
        save_dir = os.path.join(result_dir, 'images')
        util.mkdirs([result_dir, save_dir])
        loss_fn = torch.nn.MSELoss(reduction='none')
        if if_gauss:
            print('create gauss residual map')
            kp_gauss = json.load(open('./checkpoints/defect/kp_gaussion.json','r'))
            avg_pool= torch.nn.AvgPool2d((3, 3), stride=(1, 1),padding=1)
            # assume var over var_ratio*mean_var is anomaly
            var_ratio = 5
        print("start test")
        for index, data in enumerate(tqdm(dataset)):
            # data: dict_keys(['input', 'mask', 'path', 'Img', 'Ori'])
            start_time = time.time()
            output_mask, keypoint, dis_map, teacher_feats, vertex,each_kp_map,t_keypoint_list = pvnet.inference(data)
            circleed_img = util.draw_circle(data['ori_img'],t_keypoint_list)
            
            # for index_feat,teacher_feat in enumerate(teacher_feats):
            #     util.draw_CAM(torch.mean(teacher_feat,dim=1,keepdim=True), data, append_name='teacher_feat_layer_'+str(index_feat),save_count=index+1)
            
            h,w = keypoint.shape[-2],keypoint.shape[-1]
            
            # students' feature
            layer_feature_list_1 = []
            layer_feature_list_2 = []
            layer_feature_list_3 = []
            # ave students' feature
            layer_feature_list_ave = []
            for i in range(5):
                for j,student in enumerate(students):
                    eval('layer_feature_list_'+str(j+1)).append(student.get_layer_feature(data)[i])
            for i in range(5):
                layer_feature_list_ave.append((layer_feature_list_1[i]+layer_feature_list_2[i]+layer_feature_list_3[i])/self.opt.n_students)

            output_mask_ave, keypoint_ave, dis_map_ave, student_feats_ave, vertex_ave = pvnet.inference_upbone(data,layer_feature_list_ave)
            output_mask_1, keypoint_1, dis_map_1, student_feats_1, vertex_1 = pvnet.inference_upbone(data,layer_feature_list_1)
            output_mask_2, keypoint_2, dis_map_2, student_feats_2, vertex_2 = pvnet.inference_upbone(data,layer_feature_list_2)
            output_mask_3, keypoint_3, dis_map_3, student_feats_3, vertex_3 = pvnet.inference_upbone(data,layer_feature_list_3)
            # for index_feat,student1_feat in enumerate(student_feats_1):
            #     util.draw_CAM(torch.mean(student1_feat,dim=1,keepdim=True), data, append_name='student1_feat_layer_'+str(index_feat),save_count=index+1)

            t_residual_map,t_global_residual_map,s_residual_map,s_global_residual_map = util.get_vertex_loss(vertex,vertex_ave,vertex_1,vertex_2,vertex_3,h,w,keypoint,each_kp_map,loss_fn)
            
            if if_gauss:
                t_residual_map = avg_pool(t_residual_map)
                s_residual_map = avg_pool(s_residual_map)
                # new residual map for all kp zone
                s_residual_map_ = torch.zeros(s_residual_map.shape).to(device='cuda')
                t_residual_map_ = torch.zeros(t_residual_map.shape).to(device='cuda')
                each_kp_map = each_kp_map.type(torch.bool)
                for kp_index in range(self.opt.kp_num):
                    # new residual map for each kp zone
                    var_new_map = torch.zeros(s_residual_map.shape).to(device='cuda')
                    err_new_map = torch.zeros(t_residual_map.shape).to(device='cuda')
                    # neglect incomplete kp zone
                    if torch.sum(each_kp_map[kp_index])==self.opt.kp_zone_size:
                        
                        s_var = s_residual_map[0][0][each_kp_map[kp_index]]
                        # var = torch.tensor(kp_gauss['kp_'+str(kp_index)]['var']['mean']).to(device='cuda')
                        # var = torch.tensor(kp_gauss['kp_'+str(kp_index)]['var']['var']).to(device='cuda')
                        var = (s_var-torch.tensor(kp_gauss['kp_'+str(kp_index)]['var']['mean']).to(device='cuda'))**2/torch.tensor(kp_gauss['kp_'+str(kp_index)]['var']['var']).to(device='cuda')
                        var = torch.clamp(var, 0, var_ratio)
                        # assignment value for new map
                        var_new_map[0][0][each_kp_map[kp_index]]=var
                        
                        s_err = t_residual_map[0][0][each_kp_map[kp_index]]
                        # err = torch.tensor(kp_gauss['kp_'+str(kp_index)]['err']['mean']).to(device='cuda')
                        # err = torch.tensor(kp_gauss['kp_'+str(kp_index)]['err']['var']).to(device='cuda')
                        err = (s_err-torch.tensor(kp_gauss['kp_'+str(kp_index)]['err']['mean']).to(device='cuda'))**2/torch.tensor(kp_gauss['kp_'+str(kp_index)]['err']['var']).to(device='cuda')
                        err = torch.clamp(err, 0, var_ratio)
                        # assignment value for new map
                        err_new_map[0][0][each_kp_map[kp_index]]=err
                    else:
                        var_new_map[0][0][each_kp_map[kp_index]]=var_ratio
                        err_new_map[0][0][each_kp_map[kp_index]]=var_ratio
                    # get max residual for each pixel
                    s_residual_map_ = torch.maximum(s_residual_map_,var_new_map)
                    t_residual_map_ = torch.maximum(t_residual_map_,err_new_map)
                # normalize
                s_residual_map = s_residual_map_/var_ratio
                t_residual_map = t_residual_map_/var_ratio
            
            # util.draw_CAM(s_residual_map_, data, 's_var',index+1)
            # util.draw_CAM(t_residual_map_, data, 't_var',index+1)
            
            teacher_weight = 0.5
            student_weight = 0.5
            
            # torch.Size([1, 1, 384, 512]
            residual_map = teacher_weight*t_residual_map+student_weight*s_residual_map
            global_residual_map = teacher_weight*t_global_residual_map+student_weight*s_global_residual_map
            util.draw_CAM(residual_map, data, 'keypoint_residualmap',index+1)
            # util.draw_CAM(global_residual_map, data, 'global_residualmap',index+1)

            time_cost.append(time.time() - start_time)
            visuals = OrderedDict([
                                    ('input', util.tensor2im(data['input'][0], normalize=False)),
                                    ('input_mask', util.tensor2im(data['mask'][0], normalize=False)),
                                    # ('Img', util.tensor2im(data['Img'][0], normalize=False)),
                                    # ('Ori', util.tensor2im(data['Ori'][0], normalize=False)),
                                    ('generated_mask', util.tensor2im(output_mask.data[0], normalize=False)),
                                    # ('map', util.tensor2im(keypoint.data[0], normalize=False)),
                                    # ('dis_map', util.tensor2im(dis_map.data[0], normalize=False)),
                                    ('residual_map', util.tensor2im(residual_map.data[0], normalize=False)),
                                    ('circleed_img', util.tensor2im(circleed_img.data[0], normalize=False)),
                ])
            
            image_path = data['path']
            short_path = ntpath.basename(image_path[0])
            name = os.path.splitext(short_path)[0]
            image_name = '%s_%s.jpg' % (name,'residual_compare')
            save_path = os.path.join(save_dir, image_name)
            util.save_multiple_image(save_path,visuals['generated_mask'],visuals['input_mask'],visuals['residual_map'])
            for label, image_numpy in visuals.items():
                # if not label == 'residual_map' and not label == 'input_mask' and not label == 'generated_mask':
                if not label == 'generated_mask' and not label == 'input_mask':
                    image_name = '%s_%s.jpg' % (name, label)
                    save_path = os.path.join(save_dir, image_name)
                    util.save_image(image_numpy, save_path)

        time_cost = np.mean(time_cost) * 1000
        print("complete test\n")
        print(f'AVG time cost: {time_cost:4f}')

    def train_spade(self):
        print("start load opts")
        print("---------------")
        try:
            del self.opt
        except Exception as e:
            print("no existed opt")
        self.opt = SpadeOptions().parse(train=False, save=False, spade=True, students=False)
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        print("complete load opts\n")

        print("start load dataset")
        print("---------------")
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        print("complete load dataset\n")

        print("start load model")
        print("---------------")
        self.opt.model = 'pvnet'
        pvnet = create_model(self.opt)
        # show all members
        # print(dir(pvnet))
        print("complete load model\n")
    
        print("start train spade")
        print("---------------")
        os.makedirs(os.path.join('./results', 'temp'), exist_ok=True)
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
        # train_outputs = OrderedDict([('layer3_4', []), ('layer3_5', [])])
        train_feature_filepath = os.path.join('./results', 'temp', 'defect.pkl')
        train_keypoints__filepath = os.path.join('./results', 'temp', 'train_keyponits.pkl')
        # if dataset too big
        # pv_outputs = pvnet.pvnet.resnet18_8s.outputs
        # torch.tensor(pv_outputs, device='cpu')

        for i, data in enumerate(tqdm(dataset)):
            # model prediction
            # with torch.no_grad():
                # data = {key:data[key].cpu() for key in data}
            output_mask, keypoint, dis_map, _,_,each_kp_map,_ = pvnet.inference(data)
            # get intermediate layer outputs
            new_output = []
            len_ = int(len(pvnet.pvnet.resnet18_8s.outputs)/2)
            # get average for layer[-1] and layer[-2]
            for i in range(len_):
                new_output.append((pvnet.pvnet.resnet18_8s.outputs[2*i]+pvnet.pvnet.resnet18_8s.outputs[2*i+1])/2)
            for k, v in zip(train_outputs.keys(), new_output):
                # util.draw_CAM(v, data,'teacher',self.count,['layer1','layer2','layer3','layer4'])
                train_outputs[k].append(v)
            # initialize hook outputs
            pvnet.pvnet.resnet18_8s.outputs = []
            self.count = self.count + 1

        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)
            # train_outputs[k] = v.extend(v)
        
        # save extracted feature
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
        with open(train_keypoints__filepath, 'wb') as k:
            pickle.dump(keypoint, k)
        print("complete train spade")
        return train_outputs
    
    def test_teacher_upbone(self,if_gauss=False):
        print("start load opts")
        print("---------------")
        try:
            del self.opt
        except Exception as e:
            print("no existed opt")
        self.opt = TestOptions().parse(train=False, save=False, spade=False, students=False)
        self.opt.name = 'defect'
        self.opt.dataroot = os.path.join('./dataset', 'defect')
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.noise_repeat_num = 1   # test code only supports noise_repeat_num = 1
        self.opt.no_flip = True  # no flip
        print("complete load opts\n")
        
        print("start load model")
        print("---------------")
        self.opt.model = 'pvnet'
        pvnet = create_model(self.opt)
        print("complete load model\n")

        print("start load dataset")
        print("---------------")
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        print("complete load dataset\n")

        time_cost = []
        result_dir = os.path.join(self.result_path, self.name)
        save_dir = os.path.join(result_dir, 'images')
        util.mkdirs([result_dir, save_dir])
        loss_fn = torch.nn.MSELoss(reduction='none')
        if if_gauss:
            print('create gauss residual map')
            kp_gauss = json.load(open('./checkpoints/defect/kp_gaussion_mask.json','r'))
            
            avg_pool= torch.nn.AvgPool2d((3, 3), stride=(1, 1),padding=1)
            # assume var over var_ratio*mean_var is anomaly
            var_ratio = 4
        print("start test")
        for index, data in enumerate(tqdm(dataset)):
            # data: dict_keys(['input', 'mask', 'path', 'Img', 'Ori'])
            start_time = time.time()
            output_mask, keypoint, dis_map, teacher_feats, vertex,each_kp_map,t_keypoint_list = pvnet.inference(data)
            t_keypoint_list = t_keypoint_list.cpu().detach().numpy().tolist()
            anomaly_kp = t_keypoint_list.copy()
            ori_mask = output_mask.clone()
            circleed_img = util.draw_circle(data['ori_img'],t_keypoint_list)
            
            # for index_feat,teacher_feat in enumerate(teacher_feats):
            #     util.draw_CAM(torch.mean(teacher_feat,dim=1,keepdim=True), data, append_name='teacher_feat_layer_'+str(index_feat),save_count=index+1)
            
            # h,w = keypoint.shape[-2],keypoint.shape[-1]
            
            if if_gauss:
            #     print(output_mask.shape)
                # output_mask = avg_pool(output_mask)
                # new residual map for all kp zone
                output_mask_ = torch.zeros(output_mask.shape).to(device='cuda')
                each_kp_map = each_kp_map.type(torch.bool)
                index___ = 0
                for kp_index in range(self.opt.kp_num):
                    # new residual map for each kp zone
                    value_new_map = torch.zeros(output_mask.shape).to(device='cuda')
                    # neglect incomplete kp zone
                    if torch.sum(each_kp_map[kp_index])==self.opt.kp_zone_size:
                        var = torch.tensor(kp_gauss['kp_'+str(kp_index)]['value']['mean']).to(device='cuda')
                        value_new_map[0][0][each_kp_map[kp_index]] = var
                        value_new_map = value_new_map.squeeze(0).squeeze(0).detach().cpu().numpy() 
                        kp = np.asarray(t_keypoint_list[kp_index], dtype = int)
                        # kp = t_keypoint_list[kp_index].astype(int)
                        Z1 = value_new_map[kp[0]-30:kp[0]+30,kp[1]-30:kp[1]+30]
                        x = np.linspace(0,60,60,endpoint = False)
                        y = np.linspace(0,60,60,endpoint = False)
                        X, Y = np.meshgrid(x, y)
                        fig = plt.figure()
                        ax = plt.axes(projection='3d')
                        ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
                        # ax.plot_wireframe(X, Y, Z1,color = 'orange')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_zlabel('z')
                        # ax.view_init(60, 35)
                        ax.set_title('mean',fontsize=20)
                        plt.savefig('./checkpoints/defect/{}_mean_3d.jpg'.format(str(kp_index+1)))
                        
                        
                        value_new_map = torch.zeros(output_mask.shape).to(device='cuda')
                        var = torch.tensor(kp_gauss['kp_'+str(kp_index)]['value']['var']).to(device='cuda')
                        value_new_map[0][0][each_kp_map[kp_index]] = var
                        value_new_map = value_new_map.squeeze(0).squeeze(0).detach().cpu().numpy() 
                        kp = np.asarray(t_keypoint_list[kp_index], dtype = int)
                        # kp = t_keypoint_list[kp_index].astype(int)
                        Z1 = value_new_map[kp[0]-30:kp[0]+30,kp[1]-30:kp[1]+30]
                        x = np.linspace(0,60,60,endpoint = False)
                        y = np.linspace(0,60,60,endpoint = False)
                        X, Y = np.meshgrid(x, y)
                        fig = plt.figure()
                        ax = plt.axes(projection='3d')
                        ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
                        # ax.plot_wireframe(X, Y, Z1,color = 'orange')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_zlabel('z')
                        # ax.view_init(60, 35)
                        ax.set_title('variance',fontsize=20)
                        plt.savefig('./checkpoints/defect/{}_var_3d.jpg'.format(str(kp_index+1)))
                        continue
                        
                        value_ = output_mask[0][0][each_kp_map[kp_index]]
                        var = (value_-torch.tensor(kp_gauss['kp_'+str(kp_index)]['value']['mean']).to(device='cuda'))**2/torch.tensor(kp_gauss['kp_'+str(kp_index)]['value']['var']).to(device='cuda')
                        # var = (value_-torch.tensor(kp_gauss['kp_'+str(kp_index)]['value']['mean']).to(device='cuda'))**2
                        # print(torch.max(var))
                        var = var - var_ratio
                        var = var*100
                        var = torch.clamp(var, 0, 1)
                        value_new_map[0][0][each_kp_map[kp_index]]=var
                        
                        # sum mask
                        # if False:
                        if True:
                            var_ratio_2 = 20
                            sum_ = torch.tensor([torch.sum(var)]).to(device='cuda')
                            # 0.5 no use
                            # var = (sum_-torch.tensor(kp_gauss['kp_'+str(kp_index)]['sum']['mean']).to(device='cuda'))**2/torch.tensor(kp_gauss['kp_'+str(kp_index)]['sum']['var']).to(device='cuda')
                            # print(var)
                            # 1.6 little use
                            # var = sum_[0]/torch.tensor(kp_gauss['kp_'+str(kp_index)]['sum']['mean'])[0]
                            # print(var)
                            var = sum_[0] - torch.tensor(kp_gauss['kp_'+str(kp_index)]['sum']['mean'])[0]
                            # print(var)
                            # print('---')
                            # var = torch.tensor(var[0])
                            if var > var_ratio_2:
                                value_new_map[0][0][each_kp_map[kp_index]] = 1
                            else:
                                value_new_map[0][0][each_kp_map[kp_index]] = 0
                                anomaly_kp.remove(t_keypoint_list[kp_index])
                        # assignment value for new map
                    else:
                        continue
                        value_new_map[0][0][each_kp_map[kp_index]]=0
                        anomaly_kp.remove(t_keypoint_list[kp_index])
                        
                        
                    # get max residual for each pixel
                    output_mask_ = torch.maximum(output_mask_,value_new_map)
                
                exit()    
                # normalize
                output_mask = output_mask_
                
                print('-----------')

            anomaly_circleed_img = util.draw_circle(data['ori_img'],anomaly_kp)
            time_cost.append(time.time() - start_time)
            '''
            # print(keypoint.shape)
            # print(each_kp_map.shape)
            visuals = OrderedDict([
                                    # ('input', util.tensor2im(data['input'][0], normalize=False)),
                                    ('input_mask', util.tensor2im(data['mask'][0], normalize=False)),
                                    # ('Img', util.tensor2im(data['Img'][0], normalize=False)),
                                    # ('Ori', util.tensor2im(data['Ori'][0], normalize=False)),
                                    ('generated_mask', util.tensor2im(output_mask.data[0], normalize=False)),
                                    ('map', util.tensor2im(each_kp_map[0].unsqueeze(0), normalize=False)),
                                    # ('dis_map', util.tensor2im(dis_map.data[0], normalize=False)),
                                    # ('circleed_img', util.tensor2im(circleed_img[0], normalize=False)),
                                    ('anomaly_circleed_img', util.tensor2im(anomaly_circleed_img[0], normalize=False)),
                                    ('ori_mask', util.tensor2im(ori_mask[0], normalize=False)),
                ])
            
            image_path = data['path']
            short_path = ntpath.basename(image_path[0])
            name = os.path.splitext(short_path)[0]
            image_name = '%s_%s.jpg' % (name,'residual_compare')
            save_path = os.path.join(save_dir, image_name)
            util.save_multiple_image(save_path,visuals['ori_mask'],visuals['generated_mask'],visuals['input_mask'])
            for label, image_numpy in visuals.items():
                # if not label == 'residual_map' and not label == 'input_mask' and not label == 'generated_mask':
                # if not label == 'generated_mask' and not label == 'input_mask' and not label == 'ori_mask':
                image_name = '%s_%s.jpg' % (name, label)
                save_path = os.path.join(save_dir, image_name)
                util.save_image(image_numpy, save_path)
            '''
        time_cost = np.mean(time_cost) * 1000
        print("complete test\n")
        print(f'AVG time cost: {time_cost:4f}')


if __name__ == '__main__':
    Train_flag = False
    T = Train_and_Valid()

    if Train_flag:
        T.train()
        # T.train_students()
    else:
        if_gauss = True
        
        # T.train_students_gauss()
        # T.test_upbone(if_gauss)
        
        # T.test()
        
        # T.train_teacher_gauss()
        T.test_teacher_upbone(if_gauss)

