#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
# 调用opencv需要导入cv2库
# opencv是一个基于BSD许可发行的跨平台计算机视觉库
# ----由一系列c函数和少量C++类构成，提供python Ruby Matlab等语言接口，实现图像处理和计算机视觉方面的很多通用算法
import cv2
# pathlib面向对象的文件系统路径
# ----使得路径能在不同的操作平台上适用
# ----https://zhuanlan.zhihu.com/p/139783331
from pathlib import Path
# 导入拷贝库
import copy
# 日志处理库
# ----一种可以追踪某些软件运行时所发生事件的方法。
import logging
import random
# scipy.io处理mat数据
import scipy.io as sio
import matplotlib.pyplot as plt
import time
# 文件操作相关模块，可以查找符合目的的文件。
from glob import glob

import torch
import torch.nn as nn
# optim库包含多种优化算法的包，提供丰富的接口进行优化算法的调用。
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

# 30 SRM filtes
from srm_filter_kernel import all_normalized_hpf_list
# Global covariance pooling
from MPNCOV import *  # MPNCOV

# ？？这个路径放的是什么
cover_dir = '/home/ahmed/Documents/suniward0.4/base/TRN/'

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2

EPOCHS = 200
LR = 0.01

WEIGHT_DECAY = 5e-4

TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]

OUTPUT_PATH = Path(__file__).stem


# Truncation operation
# Truncation函数作为激活函数
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        # .clamp():参数为最大最小threshold
        # 将input限制在[min,max]区间中
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output


# Pre-processing Module
# 预处理模块，高通滤波器层
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        # 卷积核大小设置为5*5
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            # 如果第一个参数等于3[滤波器数量吗]，使用.pad进行填充【填充成5*5的卷积核】
            if hpf_item.shape[0] == 3:
                # constant是边缘式填充模式
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        # hpf_weight:滤波器的权重；
        # view(30,1,5,5):30个滤波器，每个滤波大小为5*5；
        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        # 滤波器：30个滤波器，每个滤波大小为5*5，填充为2
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)
    
    # 前向传播
    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)
        return output

# 整体框架
# 第一层：30个滤波的卷积，大小5*5，步幅为1，填充2
# ......
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.group1 = HPF()

        self.group2 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )

        self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)

    # 前向传播
    def forward(self, input):
        output = input

        output = self.group1(output)
        output = self.group2(output)
        output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)

        # Global covariance pooling
        output = CovpoolLayer(output)
        output = SqrtmLayer(output, 5)
        output = TriuvecLayer(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)

        return output

# 平均池化层
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 训练模型
def train(model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, sample in enumerate(train_loader):

        data_time.update(time.time() - end)

        data, label = sample['data'], sample['label']

        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        end = time.time()

        output = model(data)  # FP

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), data.size(0))

        loss.backward()  # BP
        optimizer.step()

        batch_time.update(time.time() - end)  # BATCH TIME = BATCH BP+FP
        end = time.time()

        if i % TRAIN_PRINT_FREQUENCY == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


# Adjust BN estimated value
def adjust_bn_stats(model, device, train_loader):
    model.train()

    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()

    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)

    if accuracy > best_acc and epoch > 180:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc


# Initialization
def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


# Data augmentation
class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # Rotation
        rot = random.randint(0, 3)
        data = np.rot90(data, rot, axes=[1, 2]).copy()

        # Mirroring
        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


class MyDataset(Dataset):
    def __init__(self, DATASET_DIR, partition, transform=None):
        random.seed(1234)

        self.transform = transform

        self.cover_dir = DATASET_DIR + '/cover'
        self.stego_dir = DATASET_DIR + '/stego/' + Model_NAME

        self.covers_list = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        random.shuffle(self.covers_list)
        if (partition == 0):
            self.cover_list = self.covers_list[:4000]
        if (partition == 1):
            self.cover_list = self.covers_list[4000:5000]
        if (partition == 2):
            self.cover_list = self.covers_list[5000:10000]
        assert len(self.covers_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[file_index])

        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)

        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):
    statePath = args.statePath

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    TRAIN_DATASET_DIR = args.TRAIN_DIR
    VALID_DATASET_DIR = args.VALID_DIR
    TEST_DATASET_DIR = args.TEST_DIR

    # Log files
    global Model_NAME
    Model_NAME = 'STEGO_Suniward_P0.2'  # STEGO_WOW_P0.4--STEGO_Suniward_P0.4
    info = 'AUG'  # /AUG-only
    Model_info = '/' + Model_NAME + '_' + info + '/'
    PARAMS_NAME = 'model_params.pt'
    LOG_NAME = 'model_log'
    try:
        os.mkdir(os.path.join(OUTPUT_PATH + Model_info))
    except OSError as error:
        print("Folder doesn't exists")
        x = random.randint(1, 1000)
        os.mkdir(os.path.join(OUTPUT_PATH + Model_info + str(x)))

    PARAMS_PATH = os.path.join(OUTPUT_PATH + Model_info, PARAMS_NAME)
    LOG_PATH = os.path.join(OUTPUT_PATH + Model_info, LOG_NAME)

    setLogger(LOG_PATH, mode='w')

    # Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    train_dataset = MyDataset(TRAIN_DATASET_DIR, 0, train_transform)
    valid_dataset = MyDataset(TRAIN_DATASET_DIR, 1, eval_transform)
    test_dataset = MyDataset(TRAIN_DATASET_DIR, 2, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    model = Net().to(device)
    model.apply(initWeights)

    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('-' * 8)

        all_state = torch.load(statePath)

        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']

        model.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)

        startEpoch = epoch + 1

    else:
        startEpoch = 1

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    best_acc = 0.0

    for epoch in range(startEpoch, EPOCHS + 1):
        scheduler.step()

        train(model, device, train_loader, optimizer, epoch)

        if epoch % EVAL_PRINT_FREQUENCY == 0:
            # adjust_bn_stats(model, device, train_loader)
            best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH)

    logging.info('\nTest set accuracy: \n')

    # Load best network parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    # adjust_bn_stats(model, device, train_loader)
    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH)


def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-TRAIN_DIR',
        '--TRAIN_DIR',
        help='The path to load train_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-VALID_DIR',
        '--VALID_DIR',
        help='The path to load valid_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-TEST_DIR',
        '--TEST_DIR',
        help='The path to load test_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-g',
        '--gpuNum',
        help='Determine which gpu to use',
        type=str,
        choices=['0', '1', '2', '3'],
        required=True
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = myParseArgs()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
    main(args)
