from U_net import UNet
import seaborn as sns
import os
sns.set_style('darkgrid')
import torch
import torchvision
from torchvision import transforms
from  OCTDataset import OCTDataset

def check_for_nonzero(item):
    if torch.count_nonzero(item)!=0:
        return True
    return False


# function for resizing tensors to size, size
def resize(item, size):
    T = torchvision.transforms.Resize(size=(size,size),
                                      interpolation=transforms.InterpolationMode.BILINEAR,
                                      antialias=True)
    return T(item)

if __name__ == '__main__':


    width = 284
    height = 284
    width_out = 196
    height_out = 196

    OCT = OCTDataset('2015_BOE_Chiu', None, 512)

    indices = torch.randperm(len(OCT)).tolist()
    train_indices = indices[:-8]
    test_indices = indices[-8:]
    # print(train_indices)
    # print(test_indices)
    OCTrain = torch.utils.data.Subset(OCT, train_indices)
    OCTest = torch.utils.data.Subset(OCT, test_indices)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
