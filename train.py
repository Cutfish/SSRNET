import torch
from torch import nn
from utils import to_var, batch_ids2words
import random
import torch.nn.functional as F
import cv2


def spatial_edge(x):
    """
    计算特征图的空间边缘（垂直 + 水平方向差分）。
    用于在损失函数中约束参考图 GT 与模型输出在空间边缘上的一致性。

    参数:
        x: [B, C, H, W]
    返回:
        edge1: [B, C, H-1, W]  垂直方向相邻行之差
        edge2: [B, C, H, W-1]  水平方向相邻列之差
    """
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    """
    计算特征图的光谱边缘（相邻波段之差）。
    用于在损失函数中约束参考图 GT 与模型输出在光谱边缘上的一致性，
    保持重建图像的光谱连续性。

    参数:
        x: [B, C, H, W]
    返回:
        edge: [B, C-1, H, W]  相邻波段差分
    """
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge


def train(train_list, 
          image_size, 
          scale_ratio, 
          n_bands, 
          arch, 
          model, 
          optimizer, 
          criterion, 
          epoch, 
          n_epochs):
    """
    单个 epoch 的训练过程。

    数据准备策略：
        每个 epoch 从训练区域（608×340，中心已挖空）随机裁取 image_size×image_size 的 patch，
        实现数据增强，避免每次看到相同的区域。

    损失函数（SSRNET）：
        loss = loss_fus + loss_spat_edge + loss_spec_edge
            loss_fus       = MSE(out,       image_ref)          # 重建损失
            loss_spat_edge = 0.5*MSE(edge1, ref_edge1)          # 空间边缘保持损失
                           + 0.5*MSE(edge2, ref_edge2)
            loss_spec_edge = MSE(spec_edge, ref_edge_spec)      # 光谱边缘保持损失

    参数:
        train_list : [train_ref, train_lr, train_hr] 训练集三元组
        image_size : patch 大小，默认 128
        scale_ratio: 空间下采样倍率，默认 4
        n_bands    : 全波段数量，PaviaU=103
        arch       : 架构名称
        model      : SSRNET 模型
        optimizer  : Adam 优化器
        criterion  : MSELoss
        epoch      : 当前 epoch 编号
        n_epochs   : 总 epoch 数
    """
    train_ref, train_lr, train_hr = train_list

    # ------------------------------------------------------------------ #
    # 随机裁取 image_size×image_size 的 patch，实现数据增强                 #
    # 每个 epoch 从训练区域随机取不同位置，增加样本多样性                    #
    # ------------------------------------------------------------------ #
    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h-image_size-1)
    w_str = random.randint(0, w-image_size-1)

    # 从完整训练图中裁取 patch 作为本次 epoch 的训练样本
    train_lr  = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]  # [1, 103, 128, 128]
    train_ref = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]  # [1, 103, 128, 128]
    # 重新从 train_ref patch 降采样生成 LR（相比 data_loader 里的预计算更精确）
    train_lr  = F.interpolate(train_ref, scale_factor=1/(scale_ratio*1.0))        # [1, 103, 32, 32]
    train_hr  = train_hr[:, :, h_str:h_str+image_size, w_str:w_str+image_size]   # [1, 5,  128, 128]

    model.train()

    # ------------------------------------------------------------------ #
    # 将数据移到 GPU                                                       #
    # ------------------------------------------------------------------ #
    image_lr  = to_var(train_lr).detach()   # [1, 103, 32, 32]  低分辨率全波段
    image_hr  = to_var(train_hr).detach()   # [1, 5,  128, 128] 高分辨率稀疏波段
    image_ref = to_var(train_ref).detach()  # [1, 103, 128, 128] GT（目标图）

    # ------------------------------------------------------------------ #
    # 前向传播                                                              #
    # model 返回 6 个张量：                                                 #
    #   out        [1, 103, 128, 128]  最终重建结果（SSRNET = x_spec）      #
    #   out_spat   [1, 103, 128, 128]  空间残差增强中间结果                  #
    #   out_spec   [1, 103, 128, 128]  光谱残差增强中间结果                  #
    #   edge_spat1 [1, 103, 127, 128]  输出的空间垂直边缘                    #
    #   edge_spat2 [1, 103, 128, 127]  输出的空间水平边缘                    #
    #   edge_spec  [1, 102, 128, 128]  输出的光谱边缘                       #
    # ------------------------------------------------------------------ #
    optimizer.zero_grad()
    out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = model(image_lr, image_hr)

    # 计算 GT 的空间边缘和光谱边缘（作为监督信号）
    ref_edge_spat1, ref_edge_spat2 = spatial_edge(image_ref)   # GT 空间边缘
    ref_edge_spec = spectral_edge(image_ref)                    # GT 光谱边缘

    # ------------------------------------------------------------------ #
    # 计算损失                                                              #
    # ------------------------------------------------------------------ #
    if 'RNET' in arch:
        # 重建损失：最终输出/中间结果与 GT 的 MSE
        loss_fus  = criterion(out,      image_ref)   # 最终重建损失（SSRNET/SpecRNET）
        loss_spat = criterion(out_spat, image_ref)   # 空间分支中间结果损失
        loss_spec = criterion(out_spec, image_ref)   # 光谱分支中间结果损失

        # 光谱边缘损失：约束相邻波段差分与 GT 一致，保持光谱连续性
        loss_spec_edge = criterion(edge_spec, ref_edge_spec)

        # 空间边缘损失：约束水平和垂直方向边缘与 GT 一致，保持空间锐度
        loss_spat_edge = (0.5 * criterion(edge_spat1, ref_edge_spat1)
                        + 0.5 * criterion(edge_spat2, ref_edge_spat2))

        if arch == 'SpatRNET':
            # 仅空间残差：只用空间重建损失 + 空间边缘损失
            loss = loss_spat + loss_spat_edge
        elif arch == 'SpecRNET':
            # 仅光谱残差：只用光谱重建损失 + 光谱边缘损失
            loss = loss_spec + loss_spec_edge
        elif arch == 'SSRNET':
            # 完整版：最终重建损失 + 空间边缘损失 + 光谱边缘损失
            loss = loss_fus + loss_spat_edge + loss_spec_edge
    else:
        # 其他非 RNET 架构（如 SSFCNN、TFNet 等）：仅使用简单重建损失
        loss = criterion(out, image_ref)

    # 反向传播 + 参数更新
    loss.backward()
    optimizer.step()

    # 打印当前 epoch 损失
    print('Epoch [%d/%d], Loss: %.4f'
          %(epoch, 
            n_epochs, 
            loss,
            ) 
         )
