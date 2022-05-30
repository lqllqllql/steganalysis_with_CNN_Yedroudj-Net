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
# 导入时间模块
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
# Truncation函数作为激活函数,防止更深层的数值扩大
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
# 预处理模块，高通滤波器层，提取噪声分量的残差
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
        # ----所有核的大小被设置为5*5（加权矩阵）
        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        # 滤波器：30个滤波器，每个滤波大小为5*5，填充为2
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        # 激活函数的
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

# BN:将每个特征分布归一化为零均值和单位方差，最终缩放和转换分布
# ----使得训练对初始参数不敏感，允许更大的学习率进行加速学习，提高检测精度
class AverageMeter(object):
    """computes（计算） and stores（存储） the average and current value.
     
     Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
        
        输入两个参数：loss_value:处理的数值；
                     batch_size：批次，即每个batch_size更新一次；
        本质时对所有batch_size的损失取平均。

    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # 更新
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
# 实例化Net的原因
# ----取平均，当test的batch_size过小，会被BN层导致生曾图片颜色失真过大；
# ----该函数在模型测试阶段使用；
# model=Net()
# ----https://cloud.tencent.com/developer/article/1819853

# 训练模型
# 论文信息：将所有512*512图像进行重新采样为256*256图像作为模型的输入
def train(model, device, train_loader, optimizer, epoch):
    # 取平均
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    # 计算程序运行时间
    end = time.time()
    
    # pytorch中的数据加载：下载训练集
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)
        # sample是图像吗？
        data, label = sample['data'], sample['label']

        # 数据大小：以列表的形式展示
        # data是一个256*256的图像像素的向量
        # ---将图像像素转换为list形式
        shape = list(data.size())
        # shape[0]：读取list的第一个维度的值?
        # ---shape[0] * shape[1]行，*shape[2:]列
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        # 转换成一维数组；数据标签展平（图片转换成一维行向量）
        label = label.reshape(-1)
        data, label = data.to(device), label.to(device)
        
        # 论文提到使用小批量随机梯度下降（SGD）对CNN进行训练【动量=0.95；权重衰减=0.0001】
        # zero_grad()函数:将模型的参数梯度初始化为0
        # ----【一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和】
        # backward()函数计算，当网络参量进行反馈时，梯度时累积计算而不是被替换，但在处理每一个batch时
        # ---并不需要与其他batch的梯度混合累积计算，因此需要对每个batch使用zero_grad()将参数梯度置为0；
        optimizer.zero_grad()
        # 使用cpu或gpu的时间
        end = time.time()
        # 前向传播计算预测值
        output = model(data)  # FP
        # 使用交叉熵损失函数求loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        # item():返回loss的数值
        losses.update(loss.item(), data.size(0))
        # 反向传播计算梯度
        loss.backward()  # BP
        # 更新所有参数
        optimizer.step()

        batch_time.update(time.time() - end)  # BATCH TIME = BATCH BP+FP
        end = time.time()
        
        # train_print_frequency
        if i % TRAIN_PRINT_FREQUENCY == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


# Adjust BN estimated value
def adjust_bn_stats(model, device, train_loader):
    # model.train():让model编程训练模式，启用dropout和BN，在训练中能防止网络过拟合问题；
    # ----模型中有BN和dropout训练时需要添加；
    # ----保证BN层能够用到每一批数据的均值和方差；
    # ----dropout:model.train()随机取一部分网络连接来训练更新参数
    model.train()
    # with:对资源进行访问；确保不管使用过程中是否发证异常都会执行必要的“清理”操作，释放资源
    # -----如文件使用后自动关闭\线程中锁的自动获取和释放等
    
    # tensor中的一个参数requires_grad参数：
    # ----1）True：反向传播时，tensor会自动求导（即计算梯度）
    # ----2) False:默认设置；反向传播时不会自动求导，能大大节约显存或内存
    # with torch.no_grad():在此模块下，所有计算出的tensor的requires_grad都设置为False
    # ----即所有tensor不会自动求导
    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']
            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)
            data, label = data.to(device), label.to(device)
            # 前向传播中计算预测值
            output = model(data)

# 模型的评价体系：
def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    # model.eval()：pytorch中会自动把BN和DropOut固定住不取平均（即不启用），而使用训练好的值；
    # ----保证BN层能够用全部训练数据的均值和方差，保持测试过程中BN层的均值和方差不变；
    # ----dropout：model.eval()利用了所有网络连接，不进行随机舍弃神经元
    model.eval()

    # 初始化
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']
            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)
            # data=data.to(device) :将所有读取的数据data（Tensor变量）copy一份到device所指定的GPU上，之后的运算都在GPU上运行
            data, label = data.to(device), label.to(device)

            # 计算前向传播过程的预测值
            output = model(data)
            # 按维度返回最大值【1：每一行的最大值；0：每一列的最大值】，keepdim=True判断该维度是否存在
            # ---output.max()[1]:只返回最大值的每个索引；
            # ---输出预测值最大的索引：计算正确率只需要知道样本个数即可
            pred = output.max(1, keepdim=True)[1]
            # 计算预测正确样本个数
            correct += pred.eq(label.view_as(pred)).sum().item()

    # 计算准确率
    accuracy = correct / (len(eval_loader.dataset) * 2)

    if accuracy > best_acc and epoch > 180:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    # 打印
    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc


# Initialization
# 初始化:如何进行初始化权重
def initWeights(module):
    # 如果是卷积层
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            # kaiming有正态分布和均匀分布两种：输入输出方差一致性角度考虑
            # torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
            # tensor符合正态分布或均匀分布
            # ----tensor:n维的tensor
            # ----a=0:此层后使用的rectifier的负斜率
            # ----mode:
            #     ----fan_in:保留前向传播中权重方差大小；【正向传播时，方差一致】
            #     ----fan_out:保留反向传播中权重方差大小；【反向传播时，方差一致】
            # ----nonlinerity:非线性激活函数：默认是leaky_relu
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    # 如果是全连接层
    if type(module) == nn.Linear:
        # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
        # tensor符合正态分布
        # ----tensor:n维的tensor;
        # ----mean=0:正态分布的均值
        # ----std=1:正态分布的标准差
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        # torch.nn.init.constant_(tensor, val)
        nn.init.constant_(module.bias.data, val=0)


# Data augmentation
# 数据扩充
class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # Rotation
        # random.randint():返回参数1和参数2之间的任意整数
        rot = random.randint(0, 3)
        # np.rot90(m,k=1,axes=(0,1))
        # ----在轴指定的平面中将array旋转90度
        # ----m:二维数组
        # ----k:阵列旋转90度的次数。此处是rot随机旋转rot次
        # ----axes:从1轴转到2轴
        # .cpy()：旋转后的数据复制一份
        data = np.rot90(data, rot, axes=[1, 2]).copy()

        # Mirroring
        # random.random:随机生成一个（0，1）范围内的实数；
        if random.random() < 0.5:
            # np.flip(m,axis=None)
            # ---将m在axis维进行切片，并把这个维度的元素进行颠倒
            data = np.flip(data, axis=2).copy()

        # 新增加的数据及标签
        new_sample = {'data': data, 'label': label}
        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # np.expand_dims(a,axis)
        # 插入一个新轴
        # ----axis=1:在列方向上，将数据升维：一维转换成二维
        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample

# 数据集
class MyDataset(Dataset):
    def __init__(self, DATASET_DIR, partition, transform=None):
        # 设置随机种子，方便复现结果
        random.seed(1234)

        self.transform = transform
        self.cover_dir = DATASET_DIR + '/cover'
        self.stego_dir = DATASET_DIR + '/stego/' + Model_NAME

        self.covers_list = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]、
        # 随机打乱
        random.shuffle(self.covers_list)
        # 5000张随机选4000张进行训练，剩余1000张作为验证集
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
