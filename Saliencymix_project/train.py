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
from cutout import Cutout
import cv2

import copy
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',help="write dataset you want(cifar100)",default='cifar100',type=str)
parser.add_argument('--model',help="write model you want(resnet34)",default='resnet34')
parser.add_argument('--batch_size',help='write batchsize(128)',default=128,type=int)
parser.add_argument('--epochs',help='write epoch(150)',default=150,type=int)
parser.add_argument('--learning_rate',help='write learning rate(0.1)',default=0.1,type=float)
parser.add_argument('--data_augmentation',help='write data augmentation True->1 or False->0',default=True,type=int)
parser.add_argument('--cutout',help='write cutout 1->True or 0->False',default=1,type=int)
parser.add_argument('--n_holes',help='write n_holes',default=1,type=int)
parser.add_argument('--length',help='write length',default=16,type=int)
parser.add_argument('--seed',help='write seed',default=0,type=int)
parser.add_argument('--print_freq',help="write print frequency",default=30,type=int)
parser.add_argument('--beta',help="write beta",default=1,type=int)
parser.add_argument('--salmix_prob',help='write salmix_prob',default = 0.5,type=float)
args = parser.parse_args()
dataset = args.dataset # cifar10 or cifar100
model = args.model # resnet18, resnet50, resnet101
batch_size = args.batch_size  # Input batch size for training (default: 128)
epochs = args.epochs # Number of epochs to train (default: 200)
learning_rate = args.learning_rate # Learning rate
data_augmentation = args.data_augmentation # Traditional data augmentation such as augmantation by flipping and cropping?
cutout = args.cutout # Apply Cutout?
n_holes = args.n_holes # Number of holes to cut out from image
length = args.length # Length of the holes
seed = args.seed # Random seed (default: 0)
print_freq = args.print_freq
beta = args.beta
salmix_prob = args.salmix_prob
cuda = torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

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
train_transform.transforms.append(transforms.Resize(256))
if data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(256, padding=12))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if cutout:
    train_transform.transforms.append(Cutout(n_holes=n_holes, length=length))
train_transform.transforms.append(transforms.Resize(32))

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

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

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
        #image -> input
        #labels -> target
        # measure data loading time
        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if beta > 0 and r < salmix_prob:
            lam = np.random.beta(beta,beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1,bby1,bbx2,bby2 = saliency_bbox(input[rand_index[0]],lam)
            input[:,:,bbx1:bbx2,bby1:bby2] = input[rand_index,:,bbx1:bbx2,bby1:bby2]
            lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (input.size()[-1] * input.size()[-2]))

            model.zero_grad()
            pred = model(input)
            xentropy_loss = criterion(pred,target_a)*lam + criterion(pred,target_b)*(1.-lam)
        else:
            model.zero_grad()
            pred = model(input)
            xentropy_loss = criterion(pred,target)

        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()
        # pred_max = torch.max(pred.data,1)[1]
        # total += target.size(0)
        # correct += (pred_max == target.data).sum().item()
        # accuracy = (correct/total)*100

        scheduler.step(epoch)

        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        losses.update(xentropy_loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)
    f = open('/root/volume/MidProject/Saliencymix_project/'+'train'+'.txt','a')
    save_acc = str(epoch)+'\t'+str(top1)+'\t'+str(top5)+'\n'
    f.write(save_acc)
    f.close()
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
    g = open('/root/volume/MidProject/Saliencymix_project/'+'test'+'.txt','a')
    save_acc = str(epoch)+'\t'+str(top1)+'\t'+str(top5)+'\n'
    g.write(save_acc)
    g.close()
    print('==> Test Accuracy:  Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

model = ResNet34(num_classes=num_classes).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, nesterov=True, weight_decay=5e-4)

scheduler = MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)

criterion = torch.nn.CrossEntropyLoss().cuda()

f = open('/root/volume/MidProject/Saliencymix_project/acc.txt','w')
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
    text = str(epoch)+'\t'+str(test_acc)
    f.write(text)

    elapsed_time = time.time() - start_time
    print('==> {:.2f} seconds to train this epoch\n'.format(elapsed_time))
    # learning rate scheduling
    scheduler.step()
    
    # Save model for best accuracy
    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), '/root/volume/MidProject/Saliencymix_project/model_saliency_best.pt')
f.close()
torch.save(model.state_dict(),'/root/volume/MidProject/Saliencymix_project/model_saliency_latest.pt')
print(f"Best Top-1 Accuracy: {best_acc}")