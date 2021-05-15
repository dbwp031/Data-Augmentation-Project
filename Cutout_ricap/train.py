import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from torchvision import datasets, transforms

from tqdm.notebook import tqdm as tqdm

from model import ResNet34
from utils import *

import cv2
dataset = 'cifar100' # cifar10 or cifar100
model = 'resnet34' # resnet18, resnet50, resnet101
batch_size = 128  # Input batch size for training (default: 128)
epochs = 150 # Number of epochs to train (default: 200)
learning_rate = 0.1 # Learning rate
data_augmentation = True # Traditional data augmentation such as augmantation by flipping and cropping?
RICAP = True
beta = 0.3
n_holes = 1 # Number of holes to cut out from image
length = 16 # Length of the holes
seed = 0 # Random seed (default: 0)
print_freq = 30
cuda = torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

test_id = dataset + '_' + model

# dataset = 'cifar100' # cifar10 or cifar100
# model = 'resnet34' # resnet18, resnet50, resnet101
# batch_size = 128  # Input batch size for training (default: 128)
# epochs = 150 # Number of epochs to train (default: 200)
# learning_rate = 0.1 # Learning rate
# data_augmentation = True # Traditional data augmentation such as augmantation by flipping and cropping?
# cutout = True # Apply Cutout?
# n_holes = 1 # Number of holes to cut out from image
# length = 16 # Length of the holes
# seed = 0 # Random seed (default: 0)
# print_freq = 30
# cuda = torch.cuda.is_available()
# cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

test_id = dataset + '_' + model

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)



test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

def find_nearest(array, value,W,w_,H,h_):
    array = array[:W-w_+1,:H-h_+1]
    idx = np.unravel_index((np.abs(array- value)).argmin(),array.shape)
    return idx

def saliency_bbox(img,w_,h_):
    size = img.size()
    W = size[1]
    H = size[2]

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    idx = find_nearest(saliencyMap,np.median(saliencyMap, axis=None),W,w_,H,h_)

    median_indices = idx
    x = median_indices[0]
    y = median_indices[1]

    bbx1 = x 
    bby1 = y 
    bbx2 = x + w_
    bby2 = y + h_

    return bbx1, bby1, bbx2, bby2

def ricap(beta, images, targets,):
  # size of image
  I_x, I_y = images.size()[2:]

  # generate boundary position (w, h)
  w = int(np.round(I_x * np.random.beta(beta, beta)))
  h = int(np.round(I_y * np.random.beta(beta, beta)))
  w_ = [w, I_x - w, w, I_x - w]
  h_ = [h, h, I_y - h, I_y - h]

  # select four images
  cropped_images = {}
  c_ = {}
  W_ = {}
  new_label = []

  for k in range(4):
    index = torch.randperm(images.size(0)).cuda()
    bbx1, bby1, bbx2, bby2 = saliency_bbox(images[index[0]],w_[k],h_[k])
    cropped_images[k] = images[index][:,:,bbx1:bbx2,bby1:bby2]
    c_[k] = targets[index].cuda()
    W_[k] = (w_[k] * h_[k]) / (I_x * I_y)
    soft_label = targets[index]*W_[k]  
    new_label.append(soft_label)
  
  new_label=np.asarray(new_label)
  new_label=new_label.sum(axis=0)

  # patch cropped images
  patched_images = torch.cat((torch.cat((cropped_images[0], cropped_images[1]), 2),torch.cat((cropped_images[2], cropped_images[3]), 2)),3)
  patched_images = patched_images.cuda()
  targets = (c_, W_)
  return patched_images, targets, new_label

def ricap_criterion(outputs, c_, W_):
  loss = sum([W_[k] * F.cross_entropy(outputs, Variable(c_[k])) for k in range(4)])
  return loss


xentropy_loss_avg = 0.
correct = 0.
total = 0.
    
def train(train_loader, epoch, model, optimizer, criterion):
    global xentropy_loss_avg,correct,total
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if RICAP:
          input,(c_, W_),new_label = ricap(beta,input,target)
          output = model(input)
          loss = ricap_criterion(output,c_, W_)
          acc1, acc5 = accuracy(output, new_label, topk=(1, 5))
        else:
          # compute output
          output = model(input)
          loss = criterion(output,target)
          # measure accuracy and record loss, accuracy
          acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # measure accuracy and record loss, accuracy 
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)

    print('==> Train Accuracy: Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

def test(test_loader,epoch, model):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    for i,(input,target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))
    print('==> Test Accuracy:  Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

model = ResNet34(num_classes=num_classes).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, nesterov=True, weight_decay=5e-4)

scheduler = MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)

criterion = torch.nn.CrossEntropyLoss().cuda()
###########################################################
best_acc = 0
for epoch in range(epochs):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    print("\n----- epoch: {}, lr: {} -----".format(
        epoch, optimizer.param_groups[0]["lr"]))

    # train for one epoch
    start_time = time.time()
    train(train_loader, epoch, model, optimizer, criterion)
    test_acc = test(test_loader,epoch,model)

    elapsed_time = time.time() - start_time
    print('==> {:.2f} seconds to train this epoch\n'.format(elapsed_time))
    # learning rate scheduling
    scheduler.step()
    
    # Save model for best accuracy
    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), '/root/volume/MidProject/Cutout_ricap/model_best.pt')

torch.save(model.state_dict(),'/root/volume/MidProject/Cutout_ricap/model_latest.pt')
print(f"Best Top-1 Accuracy: {best_acc}")