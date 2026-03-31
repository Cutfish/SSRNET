import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SSRNET(nn.Module):
    """
    SSRNET: Spatial-Spectral Residual Network for hyperspectral image super-resolution.

    任务目标：
        输入两路互补信息:
            x_lr  [B, n_bands,        H/r, W/r]  低分辨率全波段图（空间信息弱，光谱信息全）
            x_hr  [B, n_select_bands, H,   W  ]  高分辨率稀疏波段图（空间信息强，光谱信息少）
        重建目标:
            out   [B, n_bands,        H,   W  ]  高分辨率全波段图

    架构变体（通过 arch 参数控制）:
        SSRNET  : 空间残差 + 光谱残差（完整版，两路均有残差增强）
        SpatRNET: 仅空间残差（消融实验用）
        SpecRNET: 仅光谱残差（消融实验用）
    """

    def __init__(self, 
                 arch,
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        """
        参数:
            arch           : 架构名称，'SSRNET' / 'SpatRNET' / 'SpecRNET'
            scale_ratio    : 空间上采样倍率（LR→HR），默认 4
            n_select_bands : 高分辨率稀疏波段数量，默认 5
            n_bands        : 全波段数量，PaviaU=103
        """
        super(SSRNET, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands

        # 可学习融合权重（当前主要供扩展使用，forward 中未直接用于加权）
        self.weight = nn.Parameter(torch.tensor([0.5]))

        # ------------------------------------------------------------------ #
        # 卷积模块定义（三个模块参数独立，但结构相同）                           #
        #   conv_fus : 初步融合卷积，用于整合插值后的 LR+HR 特征               #
        #   conv_spat: 空间残差卷积，增强空间边缘细节                           #
        #   conv_spec: 光谱残差卷积，增强波段间的光谱一致性                     #
        #   均为 3×3 卷积，padding=1 保持特征图尺寸不变                        #
        #   输入/输出通道均为 n_bands（103）                                    #
        # ------------------------------------------------------------------ #
        self.conv_fus = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spat = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.conv_spec = nn.Sequential(
                  nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )

    def lrhr_interpolate(self, x_lr, x_hr):
        """
        Step 1: 将低分辨率全波段图上采样，并注入高分辨率稀疏波段，实现初步融合。

        操作流程：
            1. 对 x_lr 做双线性插值，上采样 scale_ratio 倍：
               x_lr: [B, 103, 32, 32] → [B, 103, 128, 128]
            2. 将 x_hr 中的 n_select_bands 个高分辨率波段注入对应位置：
               gap_bands ≈ 25.75（PaviaU），注入位置：波段 0, 25, 51, 77, 102
               直接替换对应通道，使网络从一开始就获得准确的空间纹理锚点

        参数:
            x_lr: [B, n_bands,        H/r, W/r]  低分辨率全波段图
            x_hr: [B, n_select_bands, H,   W  ]  高分辨率稀疏波段图
        返回:
            x:    [B, n_bands,        H,   W  ]  融合后的初始估计图
        """
        # 双线性上采样，将 LR 图放大到 HR 尺寸
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        # [B, 103, 32, 32] → [B, 103, 128, 128]

        # 计算均匀采样间隔，将高分辨率稀疏波段注入对应通道（替换双线性插值结果）
        gap_bands = self.n_bands / (self.n_select_bands-1.0)
        for i in range(0, self.n_select_bands-1):
            # 注入 x_hr 的第 i 个波段到上采样图的第 int(gap_bands*i) 个通道
            x_lr[:, int(gap_bands*i), ::] = x_hr[:, i, ::]
        # 注入最后一个波段（波段 n_bands-1 = 102）
        x_lr[:, int(self.n_bands-1), ::] = x_hr[:, self.n_select_bands-1, ::]

        return x_lr  # [B, 103, 128, 128]

    def spatial_edge(self, x):
        """
        计算空间边缘（用于空间边缘保持损失）。

        通过相邻像素做差分，提取图像在垂直和水平方向上的高频边缘信息。
        训练时约束模型输出的空间边缘与 GT 的空间边缘尽量一致，
        从而让网络学到更锐利的空间结构。

        参数:
            x: [B, C, H, W]
        返回:
            edge1: [B, C, H-1, W]  垂直方向差分（相邻行之差）
            edge2: [B, C, H, W-1]  水平方向差分（相邻列之差）
        """
        edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]   # 垂直差分
        edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]  # 水平差分

        return edge1, edge2

    def spectral_edge(self, x):
        """
        计算光谱边缘（用于光谱边缘保持损失）。

        通过相邻波段做差分，提取各像素在光谱维度上的变化信息。
        训练时约束模型输出的光谱边缘与 GT 的光谱边缘尽量一致，
        从而保持重建图像的光谱连续性和平滑性。

        参数:
            x: [B, C, H, W]
        返回:
            edge: [B, C-1, H, W]  相邻波段之差（光谱差分）
        """
        edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]
        return edge

    def forward(self, x_lr, x_hr):
        """
        前向传播。

        完整数据流（以 SSRNET + PaviaU 为例）:
        ─────────────────────────────────────────────────────────
        输入:
            x_lr  [1, 103, 32,  32 ]  低分辨率全波段
            x_hr  [1, 5,   128, 128]  高分辨率稀疏波段（5个）

        Step 1: lrhr_interpolate
            双线性上采样 x_lr → [1, 103, 128, 128]
            注入 x_hr 中5个高分辨率波段到对应通道（波段 0,25,51,77,102）
            → x: [1, 103, 128, 128]

        Step 2: conv_fus（融合卷积）
            3×3 Conv + ReLU，整合空间细节与光谱信息
            → x: [1, 103, 128, 128]

        Step 3（SSRNET 分支）:
            3a. 空间残差增强:
                x_spat = x + conv_spat(x)  ← 残差连接，保留原始信息同时增强边缘
                spat_edge1, spat_edge2 = spatial_edge(x_spat)  ← 用于空间损失监督

            3b. 光谱残差增强（以空间分支输出为输入，串联结构）:
                x_spec = x_spat + conv_spec(x_spat)  ← 残差连接
                spec_edge = spectral_edge(x_spec)    ← 用于光谱损失监督

            3c. 最终输出:
                x = x_spec  [1, 103, 128, 128]

        输出（6个张量，训练时分别参与不同损失计算）:
            x          [1, 103,     128, 128]  最终重建结果（进入 loss_fus）
            x_spat     [1, 103,     128, 128]  空间残差中间结果（进入 loss_spat）
            x_spec     [1, 103,     128, 128]  光谱残差中间结果（进入 loss_spec）
            spat_edge1 [1, 103,     127, 128]  空间垂直边缘（进入 loss_spat_edge）
            spat_edge2 [1, 103,     128, 127]  空间水平边缘（进入 loss_spat_edge）
            spec_edge  [1, 102,     128, 128]  光谱边缘（进入 loss_spec_edge）
        ─────────────────────────────────────────────────────────
        """
        # Step 1: 低分辨率上采样 + 高分辨率稀疏波段注入，得到初始融合估计
        x = self.lrhr_interpolate(x_lr, x_hr)  # [B, n_bands, H, W]

        # Step 2: 初步融合卷积，整合插值后的多通道特征
        x = self.conv_fus(x)  # [B, n_bands, H, W]

        if self.arch == 'SSRNET':
            # Step 3a: 空间残差增强（SpatRNET 子网络）
            # 残差连接：保留粗融合特征，叠加卷积学到的空间细节增量
            x_spat = x + self.conv_spat(x)  # [B, n_bands, H, W]
            # 计算空间边缘（用于损失函数中的空间边缘保持约束）
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)

            # Step 3b: 光谱残差增强（SpecRNET 子网络），串联在空间残差之后
            # 以 x_spat 为输入，在空间增强的基础上进一步增强光谱连续性
            x_spec = x_spat + self.conv_spec(x_spat)  # [B, n_bands, H, W]
            # 计算光谱边缘（用于损失函数中的光谱边缘保持约束）
            spec_edge = self.spectral_edge(x_spec)

            x = x_spec  # 最终重建结果

        elif self.arch == 'SpatRNET':
            # 消融实验：仅使用空间残差，跳过光谱残差
            x_spat = x + self.conv_spat(x)
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)
            # 光谱分支直接使用融合卷积输出（无残差增强）
            x_spec = x
            spec_edge = self.spectral_edge(x_spec)

        elif self.arch == 'SpecRNET':
            # 消融实验：仅使用光谱残差，跳过空间残差
            x_spat = x  # 空间分支直接使用融合卷积输出
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)
            # 光谱残差增强
            x_spec = x + self.conv_spec(x)
            spec_edge = self.spectral_edge(x_spec)
            x = x_spec  # 最终重建结果

        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge
