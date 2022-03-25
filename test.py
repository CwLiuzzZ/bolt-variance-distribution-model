import torch
import util.util as util
from collections import OrderedDict
from einops import rearrange, reduce
import util.global_variables as gl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import time
import numpy as np
from tqdm import trange,tqdm

def get_error_map(students_pred, teacher_pred):
    # student: (batch, student_id, h, w, vector)
    # teacher: (batch, h, w, vector)
    mu_students = reduce(students_pred, 'b id vec h w -> b vec h w', 'mean')
    err = reduce((mu_students - teacher_pred)**2, 'b vec h w -> b h w', 'sum')
    return err


def get_variance_map(students_pred):
    # student: (batch, student_id, h, w, vector)
    sse = reduce(students_pred**2, 'b id vec h w -> b id h w', 'sum')
    msse = reduce(sse, 'b id h w -> b h w', 'mean')
    mu_students = reduce(students_pred, 'b id vec h w -> b vec h w', 'mean')
    var = msse - reduce(mu_students**2, 'b vec h w -> b h w', 'sum')
    return var

# def students_test(teacher,students,dataset,device,opt):
#     # compute mean and var of teacher's outputs
#     t_mu = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])    # torch
#     t_var = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])   # torch
#     t_mu_ = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])    # torch
#     t_var_ = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])   # torch
#     N = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])   # int
#     count1 = 0

#     with torch.no_grad():
#         print('Callibrating teacher on Student dataset.')
#         for i, data in enumerate(dataset):
#             # initialize hook
#             teacher.inference(data)
#             # get intermediate layer outputs
#             teacher_outputs = util.get_layer_ave_hook_orderdict(teacher.pvnet.resnet18_8s.outputs)
#             # compute mean and var of teacher's outputs, to normalize teacher's outputs when training students
#             for k in teacher_outputs.keys():
#                 t_mu_[k], t_var_[k], N[k] = util.increment_mean_and_var(t_mu_[k], t_var_[k], N[k], teacher_outputs[k], opt.batchSize)
        
#         # t_mu.requires_grad = False
#         # make t_mu's size match teacher_outputs's size
#         for k in teacher_outputs.keys():
#             d3 = teacher_outputs[k].size()[2]
#             d4 = teacher_outputs[k].size()[3]
#             mu_array = t_mu_[k].detach().cpu().numpy()
#             var_array = t_var_[k].detach().cpu().numpy()
#             t_mu[k] = util.resize_teacher_feature(mu_array,opt.batchSize,d3,d4)
#             t_var[k] = util.resize_teacher_feature(var_array,opt.batchSize,d3,d4)

#         print('Callibrating scoring parameters on Student dataset.')
#         max_err = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])
#         max_var = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])
#         mu_err = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])
#         var_err = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])
#         N_err = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])
#         mu_var = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])
#         var_var = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])
#         N_var = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])

#         for i, data in enumerate(dataset):
#             # initialize hook
#             count1 += 1
#             teacher.inference(data)
#             # get intermediate layer outputs
#             t_out = util.get_normed_orderdict(teacher.pvnet.resnet18_8s.outputs,t_mu,t_var)
#             # util.draw_CAM(t_out, data,'teacher',count1,['layer1','layer2','layer3','layer4'])
#             students_outputs = []
#             s_out = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
#             for student in students:
#                 student(data)
#                 students_outputs.append(util.get_hook_orderdict(student.outputs))
#             for k in s_out.keys():
#                 s_out[k] = torch.stack([student_outputs[k] for student_outputs in students_outputs], dim=1)
#                 s_err = get_error_map(s_out[k], t_out[k])
#                 s_var = get_variance_map(s_out[k])
#                 mu_err[k], var_err[k], N_err[k] = util.increment_mean_and_var(mu_err[k], var_err[k], N_err[k], s_err, opt.batchSize)
#                 mu_var[k], var_var[k], N_var[k] = util.increment_mean_and_var(mu_var[k], var_var[k], N_var[k], s_var, opt.batchSize)

#                 max_err[k] = max(max_err[k], torch.max(s_err))
#                 max_var[k] = max(max_var[k], torch.max(s_var))

#         test_imgs = []
#         ori_imgs = []
#         count = 0
#         # threshold = opt.anomaly_threshold
#         # print('anomaly threshold:{}'.format(threshold))
#         time_cost = []
#         if opt.use_whole_feature:
#             print('use whloe feature map')
#         else:
#             print('use keypoint feature map')

#         for i, data in enumerate(tqdm(dataset)):
#             start_time = time.time()
#             output_mask, keypoint, dis_map = teacher.inference(data)
#             # get intermediate layer outputs
#             t_out = util.get_normed_orderdict(teacher.pvnet.resnet18_8s.outputs,t_mu,t_var)
#             # resize feature to big size
#             resized_teacher_feature = util.interpolation_test_layers(t_out)
#             # get keypoint feature
#             keypoint_teacher_feature_map = util.keypoint_feature_map(keypoint,resized_teacher_feature)
#             # resize feature to normal size
#             # keypoint_teacher_feature_map = util.interpolation_back_test_layers(t_out,keypoint_teacher_feature_map)

#             # data['Img'].shape = [1, 3, 193, 258]
#             # data: dict_keys(['input', 'mask', 'path', 'Img', 'Ori'])
#             _,_,m,n = data['Img'].shape
#             # print(data['Ori'].shape) [1, 3, 193, 258]
#             # print(data['ori_img'].shape) [1, 1544, 2064, 3]
#             test_imgs.extend(data['Img'].cpu().detach().numpy())
#             ori_imgs.extend(data['Ori'].cpu().detach().numpy())

#             score_map_list = []
#             score_maps = []
#             s_out = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
#             # avg_s_out = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
            
#             # get intermediate students' outputs
#             # whole feature
#             students_outputs = []
#             # keypoint feature
#             students_keypoint_outputs = []
#             for student in students:
#                 student(data)
#                 s_output = util.get_hook_orderdict(student.outputs)
#                 students_outputs.append(s_output)

#                 # resize feature to big size
#                 resized_feature = util.interpolation_test_layers(s_output)
#                 # get keypoint feature
#                 keypoint_feature_map = util.keypoint_feature_map(keypoint,resized_feature)
#                 # resize feature to normal size
#                 # keypoint_feature_map = util.interpolation_back_test_layers(t_out,keypoint_feature_map)
#                 students_keypoint_outputs.append(keypoint_feature_map)

#             # process feature for each layer
#             for index,k in enumerate(s_out.keys()):  # for each layer
#                 # get all students' feature
#                 if opt.use_whole_feature:
#                     s_out[k] = torch.stack([student_outputs[k] for student_outputs in students_outputs], dim=1)
#                     s_err = get_error_map(s_out[k], t_out[k])
#                 else:
#                     s_out[k] = torch.stack([student_outputs[k] for student_outputs in students_keypoint_outputs], dim=1)
#                     # s_out[k].shape [1, 3, 512, 48, 64]
#                     s_err = get_error_map(s_out[k], keypoint_teacher_feature_map[k])
#                     # s_err.shape [1, 48, 64]
#                 s_var = get_variance_map(s_out[k])
#                 score_map = (s_err - mu_err[k]) / torch.sqrt(var_err[k]) + (s_var - mu_var[k]) / torch.sqrt(var_var[k])
#                 score_map = torch.squeeze(score_map)

#                 # score_map.size() = [1, 1, 193, 258]
#                 score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=(m, n),
#                                         mode='bilinear', align_corners=False)
#                 # add weight for each layer
#                 score_maps.append(score_map*opt.loss_weight[index])

#             # average distance between the features for layers
#             # score_map.size() = [1, 193, 258]
#             score_map = (torch.mean(torch.cat(score_maps, 0), dim=0))/opt.sum_loss_weight
#             # mean_ = torch.mean(score_map)
#             # max_ = torch.max(score_map)
#             # apply gaussian smoothing on the score map
#             score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
#             mean_ = np.mean(score_map)
#             max_ = np.max(score_map)
#             #score_map.shape = (193, 258)
#             score_map_list.append(score_map)
#             time_cost.append(time.time() - start_time)
#             # visualize localization reslt
#             threshold = max_ - 0.3
#             # threshold = mean_ + 1.5
#             util.visualize_loc_result(test_imgs, ori_imgs, _, score_map_list, threshold, opt.students_save_path, count)
#             count = count + 1
#             test_imgs = []
#             ori_imgs = []
            
#         time_cost = np.mean(time_cost) * 1000
#         print(f'AVG time cost: {time_cost:4f}')