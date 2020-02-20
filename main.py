import random
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
from model_search import Network
from option.default_option import TrainOptions
import os 
import tqdm
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = TrainOptions()

train_transform, valid_transform = utils._data_transforms_xview(opt)
dataset = dset.ImageFolder(root=opt.dataset, transform=train_transform)

class_names = dataset.classes
num_classes = len(class_names)
num_train = len(dataset)
indices = list(range(num_train))

random.shuffle(indices)
random.shuffle(indices)

train_indices = indices[:int(len(indices) * opt.train_portion)]
valid_indices = indices[int(len(indices) * opt.train_portion):]

# Make seperate train and valid datasets for weighted sampling
train_set = torch.utils.data.Subset(dataset, train_indices)
valid_set = torch.utils.data.Subset(dataset, valid_indices)

######## Weighted Sampling ########

train_class_count = []
valid_class_count = []

for i in range(10):
    train_class_count += [len([x for x in train_set if x[1] == i])]
    valid_class_count += [len([x for x in valid_set if x[1] == i])]

def make_weights_for_balanced_classes(count, images, nclasses):                                                                          
    weight_per_class = [0.] * nclasses                                      
    for i in range(nclasses):                                                   
        weight_per_class[i] = 1/float(count[i])                                 
    weight = [0] * sum(count)                                            
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return torch.Tensor(weight)

train_weights = make_weights_for_balanced_classes(train_class_count, train_set, 10)                                                                
train_weights = train_weights.double()   

valid_weights = make_weights_for_balanced_classes(valid_class_count, valid_set, 10)                                                                
valid_weights = valid_weights.double()

train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(valid_weights, len(valid_weights))

##### Weighted Sampling End ######


train_queue = torch.utils.data.DataLoader(
  train_set, batch_size=opt.batch_size,
  sampler=train_sampler,
  pin_memory=True, num_workers=0)

valid_queue = torch.utils.data.DataLoader(
  valid_set, batch_size=opt.batch_size,
  sampler=valid_sampler,
  pin_memory=True, num_workers=0)

criterion = nn.CrossEntropyLoss().to(device)

model = Network(opt.init_channels, num_classes, opt.layers, criterion)
model.to(device)

optimizer_model = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
optimizer_arch = torch.optim.Adam(model.arch_parameters(), lr=opt.arch_learning_rate, betas=opt.arch_betas, weight_decay=opt.arch_weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_model, float(opt.epochs), eta_min=opt.learning_rate_min)

def train(train_queue,valid_queue, model, criterion, optimizer_arch, optimizer_model, lr_arch, lr_model):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_queue)):
      model.train()

      n = data.size(0)

      # data = Variable(data , requires_grad=True).to(device)
      # target = Variable(target, requires_grad=False).to(device, non_blocking=True)
      
      # Define Lambda to pass into equation (5)
      temperature = opt.initial_temp * np.exp(-opt.anneal_rate * batch_idx)

      optimizer_arch.zero_grad()
      optimizer_model.zero_grad()

      output = model(data , temperature)

      loss = criterion(output, target) 

      loss.backward()

      nn.utils.clip_grad_norm(model.parameters(), 5)

      optimizer_arch.step()
      optimizer_model.step()

      # Note: Using top 2 accuracy, not top 5
      prec1, prec5 = utils.accuracy(output, target, topk=(1, 2))

      objs.update(loss.data, n)
      top1.update(prec1.data , n)
      top5.update(prec5.data , n)

      if batch_idx % 500 == 0:
        print("Step : " , batch_idx , "Train_Acc_Top1 : " , top1.avg , "Train_value_loss : ", objs.avg)
  
  return top1.avg, top5.avg, objs.avg

def validate(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for batch_idx, (data, target) in tqdm.tqdm(enumerate(valid_queue)):
    # data = Variable(data, volatile=True).to(device)
    # target = Variable(target, volatile=True).to(device, non_blocking=True)

    # Define Lambda to pass into equation (5)
    temperature = opt.initial_temp * np.exp(-opt.anneal_rate * batch_idx)

    output = model(data, temperature)
    loss = criterion(output, target)

    # Note: Using top 2 accuracy, not top 5
    prec1, prec5 = utils.accuracy(output, target, topk=(1, 2))
    n = data.size(0)

    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

  return top1.avg, top5.avg, objs.avg

try:
  os.mkdir('experiments/' + str(opt.name))
except:
  pass

for epoch in range(opt.epochs):
    np.random.seed(2)
    torch.cuda.manual_seed(2)

    train_acc_top1, train_acc_top5, train_valoss = train(train_queue, valid_queue, model, criterion, optimizer_arch, optimizer_model, opt.arch_learning_rate, opt.learning_rate)

    valid_acc_top1, valid_acc_top5, valid_valoss = validate(valid_queue, model, criterion)

    print("EPOCH : ", epoch, "Train_Acc_Top1 : ", train_acc_top1, "Train_value_loss : ", train_valoss)
    print("EPOCH : ", epoch, "Val_Acc_Top1 : ", valid_acc_top1, "Loss : ", valid_valoss)

    if epoch % 10 == 0:
      torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'arch_alphas': model.arch_parameters(),
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'optimizer_arch_state_dict': optimizer_arch.state_dict(),
            'train_loss': train_valoss,
            'train_acc_top1': train_acc_top1,
            'train_acc_top5': train_acc_top5,
            'valid_loss': valid_valoss,
            'valid_acc_top1': valid_acc_top1,
            'valid_acc_top5': valid_acc_top5
          },'experiments/' + str(opt.name) + '/weights_epoch_' + str(epoch) + '.pt')