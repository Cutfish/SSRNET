from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py


def build_datasets(root, dataset, size, n_select_bands, scale_ratio):
    """
    构建训练集和测试集。
    
    策略：以原始高光谱图像的中心 size×size 区域作为测试集，
         其余区域（中心挖空）作为训练集，避免训练/测试数据泄露。
    
    输入数据来源：
        - x_lr  (low-resolution, all bands)   : 对全波段图像做高斯模糊后降采样 1/scale_ratio
        - x_hr  (high-resolution, few bands)  : 从全波段中均匀挑选 n_select_bands 个波段，保持原分辨率
        - x_ref (reference / ground-truth)    : 原始全波段全分辨率图像

    参数:
        root          : 数据集根目录
        dataset       : 数据集名称，如 'PaviaU'
        size          : 裁剪的 patch 大小（训练/测试区域的 H/W），默认 128
        n_select_bands: 高分辨率稀疏波段数量，默认 5
        scale_ratio   : 空间下采样倍率，默认 4
    
    返回:
        [train_ref, train_lr, train_hr] : 训练集三元组
        [test_ref,  test_lr,  test_hr]  : 测试集三元组
    """
    # ------------------------------------------------------------------ #
    # 1. 读取原始高光谱图像 (.mat 文件)                                    #
    #    PaviaU 原始 shape: (610, 340, 103)  [H, W, 光谱波段数]            #
    # ------------------------------------------------------------------ #
    if dataset == 'Pavia':
        img = scio.loadmat(root + '/' + 'Pavia.mat')['pavia']*1.0
    elif dataset == 'PaviaU':
        img = scio.loadmat(root + '/' + 'PaviaU.mat')['paviaU']*1.0
    elif dataset == 'Botswana':
        img = scio.loadmat(root + '/' + 'Botswana.mat')['Botswana']*1.0
    elif dataset == 'KSC':
        img = scio.loadmat(root + '/' + 'KSC.mat')['KSC']*1.0
    elif dataset == 'IndianP':
        img = scio.loadmat(root + '/' + 'Indian_pines.mat')['indian_pines_corrected']*1.0
    elif dataset == 'Washington':
        img = scio.loadmat(root + '/' + 'Washington_DC.mat')['Washington_DC']*1.0
    elif dataset == 'Urban':
        img = scio.loadmat(root + '/' + 'Urban.mat')['Y']
        img = np.reshape(img, (162, 307, 307))*1.0
        img = np.swapaxes(img, 0,2)

    print (img.shape)

    # ------------------------------------------------------------------ #
    # 2. 归一化到 [0, 255]                                                #
    # ------------------------------------------------------------------ #
    max = np.max(img)
    min = np.min(img)
    img = 255*((img - min) / (max - min + 0.0))

    # ------------------------------------------------------------------ #
    # 3. 裁掉不能被 scale_ratio 整除的边缘像素                             #
    #    目的：保证下采样后的尺寸能整除还原，避免尺寸不对齐                   #
    #    PaviaU: H=610 → 610//4*4=608，需裁掉末尾 2 行                    #
    #            W=340 → 340//4*4=340，刚好整除，w_edge=-1 表示不裁        #
    # ------------------------------------------------------------------ #
    w_edge = img.shape[0]//scale_ratio*scale_ratio-img.shape[0]
    h_edge = img.shape[1]//scale_ratio*scale_ratio-img.shape[1]
    w_edge = -1  if w_edge==0  else  w_edge   # 0 表示不需要裁，用 -1 表示取全部
    h_edge = -1  if h_edge==0  else  h_edge
    img = img[:w_edge, :h_edge, :]            # PaviaU 裁后: (608, 340, 103)

    # ------------------------------------------------------------------ #
    # 4. 计算测试区域（中心 size×size patch）的坐标                         #
    # ------------------------------------------------------------------ #
    width, height, n_bands = img.shape 
    w_str = (width - size) // 2    # 中心区域起始行
    h_str = (height - size) // 2   # 中心区域起始列
    w_end = w_str + size
    h_end = h_str + size
    img_copy = img.copy()

    # ------------------------------------------------------------------ #
    # 5. 构建测试样本                                                       #
    #                                                                      #
    #   test_ref: 中心区域原始图，shape (128, 128, 103)  ← 目标GT            #
    #   test_lr:  对 test_ref 做高斯模糊(5×5, σ=2)后                       #
    #             resize 降采样到 (32, 32, 103)  ← 低分辨率全波段输入        #
    #   test_hr:  从 103 个波段中均匀挑选 5 个波段，保持高分辨率              #
    #             gap_bands ≈ 25.75，选取波段索引: 0, 25, 51, 77, 102       #
    #             shape: (128, 128, 5)  ← 高分辨率稀疏波段输入               #
    # ------------------------------------------------------------------ #
    gap_bands = n_bands / (n_select_bands-1.0)  # 均匀采样间隔，PaviaU ≈ 25.75
    test_ref = img_copy[w_str:w_end, h_str:h_end, :].copy()   # (128, 128, 103)

    # 低分辨率图：高斯模糊模拟传感器点扩散函数，再降采样
    test_lr = cv2.GaussianBlur(test_ref, (5,5), 2)
    test_lr = cv2.resize(test_lr, (size//scale_ratio, size//scale_ratio))  # (32, 32, 103)

    # 高分辨率稀疏波段：均匀挑选 5 个波段
    test_hr = test_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1):
        test_hr = np.concatenate((test_hr, test_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    test_hr = np.concatenate((test_hr, test_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)
    # test_hr shape: (128, 128, 5)

    # ------------------------------------------------------------------ #
    # 6. 构建训练样本                                                       #
    #    将中心测试区域置 0，其余区域作为训练数据（防止数据泄露）              #
    #                                                                      #
    #   train_ref: 挖空后的完整图，shape (608, 340, 103)  ← 训练 GT         #
    #   train_lr:  高斯模糊 + 降采样，shape (152, 85, 103) ← 低分辨率输入    #
    #   train_hr:  均匀挑选 5 个波段，shape (608, 340, 5)  ← 稀疏高分辨率    #
    # ------------------------------------------------------------------ #
    img[w_str:w_end,h_str:h_end,:] = 0  # 挖空中心测试区域
    train_ref = img                      # (608, 340, 103)

    # 低分辨率图：高斯模糊 + 降采样
    train_lr = cv2.GaussianBlur(train_ref, (5,5), 2)
    train_lr = cv2.resize(train_lr, (train_lr.shape[1]//scale_ratio, train_lr.shape[0]//scale_ratio))
    # train_lr shape: (85, 152, 103)  注：cv2.resize 参数为 (W, H)

    # 高分辨率稀疏波段：均匀挑选 5 个波段
    train_hr = train_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1):
        train_hr = np.concatenate((train_hr, train_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    train_hr = np.concatenate((train_hr, train_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)
    # train_hr shape: (608, 340, 5)

    # ------------------------------------------------------------------ #
    # 7. 转换为 PyTorch Tensor，调整维度顺序 [H,W,C] → [1,C,H,W]           #
    # ------------------------------------------------------------------ #
    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)  # [1, 103, 608, 340]
    train_lr  = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0)   # [1, 103, 152, 85]
    train_hr  = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0)   # [1, 5,  608, 340]
    test_ref  = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0)   # [1, 103, 128, 128]
    test_lr   = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0)    # [1, 103, 32,  32 ]
    test_hr   = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0)    # [1, 5,  128, 128]

    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]
