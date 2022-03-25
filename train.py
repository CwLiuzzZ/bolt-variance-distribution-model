import time
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
import util.global_variables as gl
import json
from sklearn.decomposition import PCA
from data.data_loader import CreateDataLoader
from options.train_options import TrainOptions
from models.create_model import create_model
import util.util as util

# from models.students_model import student_loss
from util.visualizer import Visualizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_pvmodel(opt, dataset, visualizer, dataset_size):

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)

    opt.model = 'pvnet'
    pv_model = create_model(opt)
    optimizer = pv_model.optimizer

    for epoch in range(start_epoch, opt.niter+opt.niter_decay+1):
        
        if epoch != start_epoch:
            # epoch_iter = epoch_iter % dataset_size
            epoch_iter = 0
        for i, data in enumerate(dataset):
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            save_img = total_steps % opt.display_freq == display_delta
            
            # Forward =================
            losses, seg_mask, v_residual, kp_map = pv_model(data)

            loss_dict = OrderedDict(zip(pv_model.loss_names, losses))
            loss = loss_dict['seg_loss'] + loss_dict['vote_loss']

            # Backward ================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = OrderedDict([(k, v.data.item()) if not isinstance(v, int) else v for k, v in loss_dict.items()])          
                counter_ratio = float(epoch_iter) / dataset_size
                visualizer.print_current_errors(epoch, epoch_iter, errors)
                visualizer.plot_current_errors(epoch, errors, counter_ratio, total_steps)

            if save_img:
                visuals = OrderedDict([
                                    ('input', util.tensor2im(data['input'][0], normalize=False)),
                                    ('input_mask', util.tensor2im(data['mask'][0], normalize=False)),
                                    ('generated_mask', util.tensor2im(seg_mask.data[0], normalize=False)),
                                    ('residual', util.tensor2im(v_residual.data[0], normalize=False)),
                                    ('map', util.tensor2im(kp_map.data[0], normalize=False))
                ])
                visualizer.display_current_results(visuals, epoch, total_steps)

            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                pv_model.save('latest')      
        
        if epoch > opt.niter:
            pv_model.update_learning_rate()

        # end of epoch 
        print(f'End of epoch {epoch} / {opt.niter+opt.niter_decay} =======' )

def train_student_net(opt, teacher, student, dataset, visualizer, dataset_size,valid_dataset):
    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    epoch_loss_list = []
    valid_epoch_loss_list = []

    optimizer = student.optimizer

    for epoch in range(start_epoch, opt.niter+opt.niter_decay+1):
        loss_list = []
        if epoch != start_epoch:
            # epoch_iter = epoch_iter % dataset_size
            epoch_iter = 0
        for i, data in enumerate(dataset):
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            save_img = total_steps % opt.display_freq == display_delta
            
            # Forward =================
            _, kp_maps, _, teacher_feats,_,each_kp_map,_ = teacher.inference(data)
            losses = student(data, teacher_feats, kp_maps)

            loss_dict = OrderedDict(zip(student.loss_names, losses))
            loss = 100 * loss_dict['feat_loss']

            # Backward ================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = OrderedDict([(k, v.data.item()) if not isinstance(v, int) else v for k, v in loss_dict.items()])          
                counter_ratio = float(epoch_iter) / dataset_size
                visualizer.print_current_errors(epoch, epoch_iter, errors)
                visualizer.plot_current_errors(epoch, errors, counter_ratio, total_steps)

            if save_img:
                visuals = OrderedDict([
                                    ('input', util.tensor2im(data['input'][0], normalize=False)),
                                    # ('input_mask', util.tensor2im(data['mask'][0], normalize=False)),
                                    # ('generated_mask', util.tensor2im(seg_mask.data[0], normalize=False)),
                                    # ('residual', util.tensor2im(residual_map.data[0], normalize=False)),
                                    ('map', util.tensor2im(kp_maps.data[0], normalize=False))
                ])
                visualizer.display_current_results(visuals, epoch, total_steps)

            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                student.save('latest')   

            loss_ = float(loss.detach().cpu().numpy())
            loss_list.append(loss_/100)   
        
        valid_epoch_loss_list.append(valid_student(teacher, student, valid_dataset))
        epoch_loss_list.append(np.mean(loss_list))
        if epoch > opt.niter:
            student.update_learning_rate()

        # end of epoch 
        print(f'End of epoch {epoch} / {opt.niter+opt.niter_decay} =======' )
    
        # save epoch ave loss to local
        loss_save_path = os.path.join(opt.checkpoints_dir, opt.name, 'student_'+str(student.id)+'_epoch_ave_loss'+'.png')
        np.save(os.path.join(opt.checkpoints_dir, opt.name, 'student_'+str(student.id)+'train_loss.npy'),np.array(epoch_loss_list))
        np.save(os.path.join(opt.checkpoints_dir, opt.name, 'student_'+str(student.id)+'valid_loss.npy'),np.array(valid_epoch_loss_list))
        util.draw_array_png(np.array(epoch_loss_list),np.array(valid_epoch_loss_list),loss_save_path,'epoch','loss','epoch_ave_loss')

def cal_kpZones_loss_gauss_distr(opt,teacher,students,train_dataset):
    with torch.no_grad():
        avg_pool= torch.nn.AvgPool2d((3, 3), stride=(1, 1),padding = 1)
        loss_fn = torch.nn.MSELoss(reduction='none')
        err_list = []
        var_list = []
        kp_gauss = {}
        for kp_index in range(opt.kp_num):
            kp_gauss['kp_'+str(kp_index)] = {'err':{'mean':0,'var':0},'var':{'mean':0,'var':0}}
            err_list.append([])
            var_list.append([])
        # for i, data in enumerate(tqdm(train_dataset)):
        for i, data in enumerate(train_dataset):
            # Forward =================
            output_mask, keypoint, dis_map, teacher_feats, vertex,each_kp_map,keypoint_list = teacher.inference(data)
            h,w = keypoint.shape[-2],keypoint.shape[-1]
            
            # ####
            # keypoint_gt = data['cpoints'].squeeze(0).squeeze(0)
            # disparity = []
            # for index_kp in range(keypoint_gt.shape[0]):
            #     ans = torch.abs(keypoint_gt[index_kp]- keypoint_list[index_kp])
            #     ans = float(torch.sum(ans).detach().cpu().numpy())
            #     # print(ans)
            #     disparity.append(ans) 
            # print('max',max(disparity))
            # util.kp_circle_on_image_lwe(data,str(max(disparity)), keypoint_gt,keypoint_list)
            # print('------------------------')
            # continue
            # ####
            
            # students' feature
            layer_feature_list_1 = []
            layer_feature_list_2 = []
            layer_feature_list_3 = []
            # ave students' feature
            layer_feature_list_ave = []
            
            
            for m in range(5):
                for n,student in enumerate(students):
                    eval('layer_feature_list_'+str(n+1)).append(student.get_layer_feature(data)[m])
            for m in range(5):
                layer_feature_list_ave.append((layer_feature_list_1[m]+layer_feature_list_2[m]+layer_feature_list_3[m])/opt.n_students)

            _, _, _, _, vertex_ave = teacher.inference_upbone(data,layer_feature_list_ave)
            _, _, _, _, vertex_1 = teacher.inference_upbone(data,layer_feature_list_1)
            _, _, _, _, vertex_2 = teacher.inference_upbone(data,layer_feature_list_2)
            _, _, _, _, vertex_3 = teacher.inference_upbone(data,layer_feature_list_3)
            
            t_residual_map,t_global_residual_map,s_residual_map,s_global_residual_map = util.get_vertex_loss(vertex,vertex_ave,vertex_1,vertex_2,vertex_3,h,w,keypoint,each_kp_map,loss_fn)
            t_residual_map = avg_pool(t_residual_map)
            s_residual_map = avg_pool(s_residual_map)
            
            each_kp_map = each_kp_map.type(torch.bool)
            for kp_index in range(opt.kp_num):
                if torch.sum(each_kp_map[kp_index])==opt.kp_zone_size:
                    s_err = t_residual_map[0][0][each_kp_map[kp_index]]
                    err_list[kp_index].append(s_err)
                    s_var = s_residual_map[0][0][each_kp_map[kp_index]]
                    var_list[kp_index].append(s_var)
                    
        for kp_index in range(opt.kp_num):
            err = torch.stack(err_list[kp_index])
            mean,var = util.get_mean_var(err,0)
            kp_gauss['kp_'+str(kp_index)]['err']['mean']= mean.cpu().detach().numpy().tolist()
            kp_gauss['kp_'+str(kp_index)]['err']['var']= var.cpu().detach().numpy().tolist()
            var = torch.stack(var_list[kp_index])
            mean,var = util.get_mean_var(var,0)
            kp_gauss['kp_'+str(kp_index)]['var']['mean']= mean.cpu().detach().numpy().tolist()
            kp_gauss['kp_'+str(kp_index)]['var']['var']= var.cpu().detach().numpy().tolist()
        json.dump(kp_gauss, open('./checkpoints/defect/kp_gaussion.json','w'))
        
def valid_student(teacher, student, valid_dataset):
    with torch.no_grad():
        loss_list = []
        for i, data in enumerate(valid_dataset):
            # Forward =================
            _, kp_maps, _, teacher_feats,_,each_kp_map,_ = teacher.inference(data)
            losses = student(data, teacher_feats, kp_maps)

            loss_dict = OrderedDict(zip(student.loss_names, losses))
            loss = loss_dict['feat_loss']
            loss_ = float(loss.detach().cpu().numpy())
            loss_list.append(loss_)   
        return np.mean(loss_list)

# def train_students_nets(opt, teacher, students, dataset):
#     # Choosing device 
#     device = torch.device("cuda" if opt.gpu_ids else "cpu")
#     print(f'Device used: {device}')

#     # Define optimizer
#     optimizers = [optim.Adam(student.parameters(), 
#                             lr=opt.lr, 
#                             # betas=(opt.beta1, 0.999),
#                             weight_decay=opt.weight_decay) for student in students]


#     # compute mean and var of teacher's outputs
#     t_mu = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])    # torch
#     t_var = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])   # torch
#     t_mu_ = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])    # torch
#     t_var_ = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])   # torch
#     N = OrderedDict([('layer1', 0), ('layer2', 0), ('layer3', 0), ('layer4', 0)])   # int
#     # Compute incremental mean and var over traininig set
#     # because the whole training set takes too much memory space 
#     with torch.no_grad():
#         for i, data in enumerate(dataset):
#             # Forward =================
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

#     # init SummaryWriter
#     writer_list = []
#     for j, student in enumerate(students):
#         writer_list.append(SummaryWriter('./results/runs'))
#         min_running_loss = np.inf
#         path = './checkpoints/defect/students'
#         if not os.path.exists(path):
#             os.mkdir(path)
#         model_name = f'./checkpoints/defect/students/student_{opt.patch_size}_net_{j}.pt'
#         print(f'Training Student {j} on anomaly-free dataset ...')

#         for epoch in range(opt.max_epoch):
#             running_loss = 0.0
#             epoch_loss = 0.0

#             for i, data in enumerate(dataset):
#                 # zero the parameters gradient
#                 optimizers[j].zero_grad()

#                 # Forward =================
#                 # get intermediate layer outputs
#                 with torch.no_grad():
#                     teacher.inference(data)
#                 teacher_outputs = util.get_normed_orderdict(teacher.pvnet.resnet18_8s.outputs,t_mu,t_var)
#                 student(data)
#                 students_outputs = util.get_hook_orderdict(student.outputs)
#                 # students_outputs.requires_grad = True

#                 loss = student_loss(students_outputs, teacher_outputs,opt)

#                 # backward pass
#                 loss.backward()
#                 optimizers[j].step()
#                 running_loss += loss.item()
#                 epoch_loss += loss.item()

#                 # print stats
#                 if i % 10 == 9:
#                     print(f"Epoch {epoch+1}, iter {i+1} \t loss: {running_loss}")
                    
#                     if running_loss < min_running_loss and epoch > 0:
#                         torch.save(student.state_dict(), model_name)
#                         print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
#                         print(f"Model saved to {model_name}.")

#                     min_running_loss = min(min_running_loss, running_loss)
#                     running_loss = 0.0
            
#             writer_list[j].add_scalar('student_'+str(j)+'_Train/Loss', epoch_loss / 100, epoch)
#             epoch_loss = 0.0
#         writer_list[j].close()
#             # print('complete student %d epoch %d)' % (j,epoch))

def cal_mask_gauss_distr(opt,teacher,train_dataset):
    with torch.no_grad():
        avg_pool= torch.nn.AvgPool2d((3,3), stride=(1, 1),padding = 1)
        loss_fn = torch.nn.MSELoss(reduction='none')
        value_list = []
        sum_list = []
        kp_gauss = {}
        right = 0
        wrong = 0
        # for t in range(11):
        #     print(kp_gauss['kp_'+str(t)]['sum'])
        for kp_index in range(opt.kp_num):
            kp_gauss['kp_'+str(kp_index)] = {'value':{'mean':0,'var':0},'sum':{'mean':0,'var':0}}
            value_list.append([])
            sum_list.append([])
        # kp_gauss = json.load(open('./checkpoints/defect/kp_gaussion_mask.json','r'))
            
        for i, data in enumerate(tqdm(train_dataset)):
            # Forward =================
            output_mask, keypoint, dis_map, teacher_feats, vertex,each_kp_map,keypoint_list = teacher.inference(data)
            h,w = keypoint.shape[-2],keypoint.shape[-1]
            
            # ####
            # keypoint_gt = data['cpoints'].squeeze(0).squeeze(0)
            # disparity = []
            # for index_kp in range(keypoint_gt.shape[0]):
            #     ans = torch.abs(keypoint_gt[index_kp]- keypoint_list[index_kp])
            #     ans = float(torch.sum(ans).detach().cpu().numpy())
            #     # print(ans)
            #     disparity.append(ans) 
            # print('max',max(disparity))
            # util.kp_circle_on_image_lwe(data,str(max(disparity)), keypoint_gt,keypoint_list)
            # print('------------------------')
            # continue
            # ####

            output_mask = avg_pool(output_mask)
            each_kp_map = each_kp_map.type(torch.bool)
            for kp_index in range(opt.kp_num):
                if torch.sum(each_kp_map[kp_index])==opt.kp_zone_size:
                    right += 1
                    value_err = output_mask[0][0][each_kp_map[kp_index]]
                    value_list[kp_index].append(value_err)
                else:
                    wrong += 1
                    
        for kp_index in range(opt.kp_num):
            err = torch.stack(value_list[kp_index])
            mean,var = util.get_mean_var(err,0)
            kp_gauss['kp_'+str(kp_index)]['value']['mean']= mean.cpu().detach().numpy().tolist()
            kp_gauss['kp_'+str(kp_index)]['value']['var']= var.cpu().detach().numpy().tolist()
            
        for i, data in enumerate(tqdm(train_dataset)):
            # Forward =================
            output_mask, keypoint, dis_map, teacher_feats, vertex,each_kp_map,keypoint_list = teacher.inference(data)                      
            output_mask = avg_pool(output_mask)
            # new residual map for all kp zone
            each_kp_map = each_kp_map.type(torch.bool)
            for kp_index in range(opt.kp_num):
                # neglect incomplete kp zone
                if torch.sum(each_kp_map[kp_index])==opt.kp_zone_size:
                    value_ = output_mask[0][0][each_kp_map[kp_index]]
                    var = (value_-torch.tensor(kp_gauss['kp_'+str(kp_index)]['value']['mean']).to(device='cuda'))**2/torch.tensor(kp_gauss['kp_'+str(kp_index)]['value']['var']).to(device='cuda')
                    # print(torch.max(var))
                    # print(torch.mean(var))
                    # print(torch.sum(var))
                    var = var - 4
                    var = var*100
                    var = torch.clamp(var, 0, 1)
                    sum_list[kp_index].append(torch.tensor([torch.sum(var)]))
                    # print(torch.sum(var))
                    # print('----')
        # print(sum_list)
        for kp_index in range(opt.kp_num):
            err = torch.stack(sum_list[kp_index])
            mean,var = util.get_mean_var(err,0)
            kp_gauss['kp_'+str(kp_index)]['sum']['mean']= mean.cpu().detach().numpy().tolist()
            kp_gauss['kp_'+str(kp_index)]['sum']['var']= var.cpu().detach().numpy().tolist()
        
        for t in range(11):
            print(kp_gauss['kp_'+str(t)]['sum'])
        json.dump(kp_gauss, open('./checkpoints/defect/kp_gaussion_mask.json','w'))
        print('right:',right,' wrong:',wrong)