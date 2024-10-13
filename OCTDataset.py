import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm as tqdm
import torchvision
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import math
import random
import os
import scipy.io as io
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
input_path = os.path.join('2015_BOE_Chiu')
subject_path = [os.path.join(input_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)] + [
    os.path.join(input_path, 'Subject_10.mat')]
data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
# data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]

# width = 284
# height = 284
# width_out = 196
# height_out = 196
def check_for_nonzero(item):
    if torch.count_nonzero(item)!=0:
        return True
    return False


# function for resizing tensors to size, size
def resize(item,size):
    T = torchvision.transforms.Resize(size=(size,size),
                                      interpolation=transforms.InterpolationMode.BILINEAR,
                                      antialias=True)
    return T(item)

class OCTDataset(Dataset):
    def __init__(self, root, transforms, size):
        self.root = root
        self.transforms = transforms
        self.subject_path = [os.path.join(self.root, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)] + [
            os.path.join(input_path, 'Subject_10.mat')]

        self.size = size
        self.images = torch.tensor([])
        self.masks = torch.tensor([])
        self.y = []

        self.load_images_and_masks()

        print(self.images.size(), self.masks.size())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        obj_ids = mask.unique()
        obj_ids = obj_ids[1:]

        num_objs = len(obj_ids)

        # one hot encoding masks
        masks = mask == obj_ids[:, None, None]

        boxes = []

        for i in range(num_objs):
            pos = torch.nonzero(masks[i])
            #             print(pos)
            mins, _ = torch.min(pos, dim=0)
            maxs, _ = torch.max(pos, dim=0)

            #             print(pos)
            xmin = mins[1]
            ymin = mins[0]

            xmax = maxs[1]
            ymax = maxs[0]

            boxes.append([xmin, ymin, xmax, ymax])

        # mrcnn only needs boxes, labels, and masks for the target
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

        pass

    def load_images_and_masks(self):

        # for all images
        for i in tqdm(range(len(self.subject_path))):  # range(len(self.subject_path))

            mat = io.loadmat(self.subject_path[i])

            images = np.expand_dims(np.transpose(mat['images'], (2, 0, 1)) / 255, 0)
            y = np.transpose(mat['manualFluid1'], (2, 0, 1))
            masks = np.expand_dims(np.nan_to_num(y), 0)
            #             print(masks.shape)

            # convert to tensor
            images = torch.as_tensor(images, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            images = resize(images, self.size)
            masks = resize(masks, self.size)

            #             print(masks.size())

            # we only keep images and masks with non-zero values
            for idx in range(images.shape[1]):

                mask = masks[0][idx]
                if check_for_nonzero(mask):
                    temp1 = images[::, idx, ::]

                    temp2 = masks[0, idx, ::].unsqueeze(0)

                    # make image 3 channel instead of 1 -> (1, 3, H, W)
                    img = torch.cat([temp1] * 3).unsqueeze(0)
                    #                     print(img.shape)

                    self.images = torch.cat((self.images, img))
                    self.masks = torch.cat((self.masks, temp2))
