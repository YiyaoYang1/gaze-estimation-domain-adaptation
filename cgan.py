import torch
import time
import sys
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feature = out.reshape(x.shape[0], -1)
        out = self.fc(feature)
        return out, feature


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 512)
        out = self.model(x)
        return out


class CDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        for i in range(1, 10):
            csv_file2 = csv_file + "/000%s/face" % (str(i))
            tmp = glob.glob(os.path.join(csv_file2, '*.jpg'))
            self.data += tmp
        for i in range(10, 57):
            csv_file2 = csv_file + "/00%s/face" % (str(i))
            tmp = glob.glob(os.path.join(csv_file2, '*.jpg'))
            self.data += tmp
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = []
        image_name = self.data[idx]
        image_tmp = cv2.imread(image_name)
        image = torch.from_numpy(image_tmp)
        image = image.transpose(0, 2).contiguous()
        label_tmp = image_name.split("\'")
        label_tmp = label_tmp[-1].split("_")
        label1 = int(label_tmp[3].strip("V"))
        label2 = int(label_tmp[2].strip("P")) + int(label_tmp[4].strip("H.jpg"))
        label.append(label1 * np.pi / 180)
        label.append(label2 * np.pi / 180)
        label = torch.tensor(label)
        type_ = torch.tensor([0, 1])  # fake image
        type_g = torch.tensor([1, 0])
        return image, label, type_, type_g


class MPIIDataset(Dataset):
    def __init__(self, mpii_file, transform=None):
        for k in range(2):
            mpii_file2 = mpii_file + "/p0%s.label" % (str(k))
            tmp = pd.read_csv(mpii_file2, sep=' ')
            tmp = tmp[['Face', '2DGaze']]
            if k == 0:
                self.data = tmp
            else:
                self.data = pd.concat([self.data, tmp])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]
        image_path = 'datasets\\MPIIFaceGaze\\Image\\' + image_name
        image_tmp = cv2.imread(image_path)
        image_tmp = torch.from_numpy(image_tmp)
        image = image_tmp.transpose(0, 2).contiguous()
        label = self.data.iloc[idx, 1].split(',')
        label = [float(x) for x in label]
        label = torch.tensor(label)
        type_ = torch.tensor([1, 0])  # real image
        type_g = torch.tensor([0, 1])
        return image, label, type_, type_g


if __name__ == '__main__':
    annotations_file1 = r"datasets\ColumbiaGazeCutSet"
    train_datasetC = CDataset(csv_file=annotations_file1)

    annotations_file = "datasets/MPIIFaceGaze/Label"
    train_datasetMPII = MPIIDataset(mpii_file=annotations_file)

    train_dataset = torch.utils.data.ConcatDataset([train_datasetC, train_datasetMPII])
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=8, num_workers=8)
    train_dataloaderC = DataLoader(train_datasetC, batch_size=50, shuffle=True, pin_memory=True,
                                   persistent_workers=True, prefetch_factor=8, num_workers=8)
    
    device = torch.device("cuda:0")
    lossfunc = "MSELoss"
    loss_op = getattr(nn, lossfunc)().cuda()
    loss2 = nn.BCELoss().cuda()
    savepath = r"E:\yuusa\code\my_mpiifacegaze\saveGAN"
    Discriminator = Discriminator()
    Discriminator.load_state_dict(torch.load(os.path.join(savepath, f"Iter_2_discriminator.pt")))
    myResnet = ResNet18()
    myResnet.load_state_dict(torch.load(os.path.join(savepath, f"Iter_10_model.pt")))
    myResnet.train()
    Discriminator.train()
    d_optimizer = optim.Adam(Discriminator.parameters(), lr=1e-5, betas=(0.9, 0.95))
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=237 * 5, gamma=0.1)
    g_optimizer = optim.Adam(myResnet.parameters(), lr=1e-5, betas=(0.9, 0.95))
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=237 * 3 * 3, gamma=0.1)
    Discriminator.to(device)
    myResnet.to(device)
    myepoch = 10

    for epoch in range(1, myepoch + 1):
        for i, (data, label, type_, type_g) in enumerate(train_dataloader):
            data = data.to(device)
            type_ = type_.to(device)
            with torch.no_grad():
                gaze, feature = myResnet(data.to(torch.float32))
            disc = Discriminator(feature)
            d_loss = loss2(disc.float(), type_.float())
            print("epoch %d, time %d, d_loss %f" % (epoch, i, d_loss))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            d_scheduler.step()
        torch.save(Discriminator.state_dict(), os.path.join(savepath, f"Iter_{epoch}_discriminator.pt"))
        print("Save Model Iter_%d_discriminator.pt" % (epoch))

        for i, (data, label, type_, type_g) in enumerate(train_dataloaderC):
            data = data.to(device)
            type_g = type_g.to(device)

            gaze, feature = myResnet(data.to(torch.float32))
            disc = Discriminator(feature)
            g_loss = loss2(disc.float(), type_g.float())

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            g_scheduler.step()
            print("epoch %d, time %d, g_loss %f" % (epoch, i, g_loss))
        torch.save(myResnet.state_dict(), os.path.join(savepath, f"Iter_{epoch}_Resnet.pt"))
        print("Save Model Iter_%d_%s.pt" % (epoch, 'Resnet'))


