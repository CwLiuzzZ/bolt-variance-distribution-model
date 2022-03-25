import torch
from .models import PVModel, StudentModel
import os


class FullModel(torch.nn.Module):
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, targets, *inputs):
    outputs = self.model(*inputs)
    loss = self.loss(outputs, targets)
    return torch.unsqueeze(loss,0),outputs

def DataParallel_withLoss(model,loss,**kwargs):
    model=FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None
    if 'output_device' in kwargs.keys():
        output_device=kwargs['output_device']
    else:
        output_device=None
    if 'cuda' in kwargs.keys():
        cudaID=kwargs['cuda'] 
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
    return model

def create_model(opt):
    if opt.model == 'pvnet': 
        model = PVModel()
    elif opt.model == 'student':
        model = StudentModel()
    else:
        raise(f'No {opt.model} exsit!')

    model.initialize(opt)
    if opt.DataParallel:
        print("use torch data parallel")
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        return model.module
    else:
        return model
