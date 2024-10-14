from U_net import UNet
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm as tqdm
import os
import scipy
import numpy as np
import torch
import torchvision
from torchvision import transforms
from OCTDataset import OCTDataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange
from time import sleep
import torch.nn.functional as F
import sys
import cv2
import wandb
import pickle

from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample = self.transform(x_sample)
            y_sample = self.transform(y_sample)

        return x_sample, y_sample


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = 512
    height = 1000
    width_out = 512
    height_out = 1000
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-Unet-project",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "architecture": "UNet",
            "epochs": 10,
            "batch_size": 7,
        }
    )
    torch.cuda.empty_cache()
    input_path = os.path.join('data')
    subject_folder = os.path.join(input_path, 'image')
    label_folder = os.path.join(input_path, 'label')
    x = []
    for path in os.listdir(subject_folder)[:200]:
        image = cv2.imread(os.path.join(subject_folder, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = resize(image, (width, width))
        x.append(image)
    y = []
    for path in os.listdir(label_folder)[:200]:
        image = cv2.imread(os.path.join(label_folder, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = resize(image, (width, width))
        y.append(image)
    x = torch.tensor(np.array(x))
    y = torch.tensor(np.array(y))
    x = x.view(-1, 1, x.shape[1], x.shape[2]).float()
    y = y.view(-1, 1, y.shape[1], y.shape[2]).long()
    train_dataset = CustomDataset(x[:180], y[:180])
    test_dataset = CustomDataset(x[180:], y[180:])
    batch_size = 2
    n_classes = 3
    n_channels = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_dataloader):
            # 如果你需要将数据移动到GPU，请取消注释以下两行
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs.resize(batch_size * width_out * height_out, n_classes)
            labels = labels.resize(batch_size * width_out * height_out)

            # 计算损失
            loss = criterion(outputs.float(), labels.long())  # 确保 labels 是长整型

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()
            wandb.log({"loss": loss})
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
    save_path = os.path.join("saved models", "test_version_unet1.pth")
    torch.save(model.state_dict(), save_path)
    wandb.finish()

    model = UNet(n_channels=n_channels, n_classes=n_classes)
    model_state = torch.load("saved models/test_version_unet1.pth")
    model.load_state_dict(model_state)
    out_images = []
    model.eval()
    for i, (inputs, labels) in enumerate(test_dataloader):
        outputs = model(inputs).detach().numpy()
        outputs = np.transpose(outputs[0], (1, 2, 0))
        out_images.append(outputs)
    out_images = np.array(out_images)
    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(10, 50))

    for i in range(len(test_dataloader)):
        # 显示第一组图片
        axes[i, 0].imshow(y[i, 0])
        # axes[i, 0].axis('off')
        # 显示第二组图片
        axes[i, 1].imshow(out_images[i].argmax(axis=2))
        # axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


