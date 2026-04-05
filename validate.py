from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils import *
import cv2
import pdb
from datetime import datetime
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam


def validate(test_list, arch, model, epoch, n_epochs, writer=None, dataset='dataset', log_file=None):
    """
    在测试集上执行一次推理，计算各项图像质量指标并记录日志。

    测试集数据（test_list）来源于原始图像中心的 128×128 区域，从未参与训练。

    评估指标:
        RMSE  : 均方根误差（越小越好，反映像素级重建精度）
        PSNR  : 峰值信噪比（越大越好，单位 dB）
        ERGAS : 光谱全局相对误差（越小越好，反映光谱保真度）
        SAM   : 光谱角度图（越小越好，反映光谱形状相似度，单位 rad 或 degree）

    参数:
        test_list : [test_ref, test_lr, test_hr] 测试集三元组
        arch      : 架构名称
        model     : 已训练（或训练中）的模型
        epoch     : 当前 epoch 编号（用于日志记录）
        n_epochs  : 总 epoch 数
        writer    : TensorBoard SummaryWriter（可选）
        dataset   : 数据集/架构标签（用于 TensorBoard 分组）
        log_file  : 指标记录文件路径（每 epoch 追加写入）

    返回:
        psnr (float): 当前 epoch 的 PSNR 值（用于判断是否保存最优模型）
    """
    test_ref, test_lr, test_hr = test_list
    model.eval()  # 切换到推理模式（关闭 Dropout/BatchNorm 的训练行为）

    psnr = 0
    with torch.no_grad():  # 关闭梯度计算，节省显存和计算
        # 将测试数据移到 GPU
        ref = to_var(test_ref).detach()  # [1, 103, 128, 128]  GT 参考图
        lr  = to_var(test_lr).detach()   # [1, 103, 32,  32 ]  低分辨率全波段
        hr  = to_var(test_hr).detach()   # [1, 5,   128, 128]  高分辨率稀疏波段

        # 前向推理，取第一个返回值（最终重建结果）
        # SSRNET / 其他架构统一取 out（第一个输出）
        if arch == 'SSRNet':
            out, _, _, _, _, _ = model(lr, hr)   # 取最终融合输出
        elif arch == 'SSRSpat':
            _, out, _, _, _, _ = model(lr, hr)   # 取空间残差分支输出
        elif arch == 'SSRSpec':
            _, _, out, _, _, _ = model(lr, hr)   # 取光谱残差分支输出
        else:
            out, _, _, _, _, _ = model(lr, hr)   # 默认取最终融合输出

        # 转回 CPU numpy，用于指标计算
        ref = ref.detach().cpu().numpy()   # [1, 103, 128, 128]
        out = out.detach().cpu().numpy()   # [1, 103, 128, 128]

        # ------------------------------------------------------------------ #
        # 计算各项图像质量评估指标                                              #
        # ------------------------------------------------------------------ #
        rmse  = calc_rmse(ref, out)   # 均方根误差
        psnr  = calc_psnr(ref, out)   # 峰值信噪比
        ergas = calc_ergas(ref, out)  # 光谱全局相对误差
        sam   = calc_sam(ref, out)    # 光谱角度图

        # ------------------------------------------------------------------ #
        # 将指标记录到 txt 文件（每个 epoch 追加一行）                           #
        # 格式: epoch, rmse, psnr, ergas, sam                                #
        # ------------------------------------------------------------------ #
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{dataset}_metrics_{timestamp}.txt'
        
        with open(log_file, 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) + ',' + '\n')
        
        # ------------------------------------------------------------------ #
        # 每 100 个 epoch 向 TensorBoard 写入一次（避免日志文件过大）           #
        # ------------------------------------------------------------------ #
        if epoch % 100 == 0:
            if writer:
                writer.add_scalar(f'metrics/{dataset}/RMSE',  rmse,  epoch)
                writer.add_scalar(f'metrics/{dataset}/PSNR',  psnr,  epoch)
                writer.add_scalar(f'metrics/{dataset}/ERGAS', ergas, epoch)
                writer.add_scalar(f'metrics/{dataset}/SAM',   sam,   epoch)

    return psnr
