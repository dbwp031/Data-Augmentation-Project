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

class Grayt(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self):
        return

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        r = np.random.rand(1)
        if r>0.5:
            return transforms.Grayscale(3)(img)
        else:
            return img

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if cutout:
    train_transform.transforms.append(Cutout(n_holes=n_holes, length=length))

test_transform = transforms.Compose([
    Grayt(),transforms.ToTensor(),
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


def train(train_loader, epoch, model, optimizer, criterion):
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

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

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
        torch.save(model.state_dict(), '/root/volume/MidProject/Basic/model_best_sg.pt')

torch.save(model.state_dict(),'/root/volume/MidProject/Basic/model_latest_sg.pt')
print(f"Best Top-1 Accuracy: {best_acc}")