import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import datasets, transforms
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]


        makeGray = transforms.Grayscale(num_output_channels=3)
        for n in range(self.n_holes):
            graymask = np.zeros((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            graymask[y1: y2, x1: x2] = 1

            graymask = torch.from_numpy(graymask)
            graymask = graymask.expand_as(img)
            grayimg = img * graymask
            grayimg = makeGray(grayimg)
            img[:,y1:y2,x1:x2]=grayimg[:,y1:y2,x1:x2]
        return img
