import time
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F


args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 0

print (args)


def main():
    # ------------------------------------------------------------------ #
    # 1. 数据集构建                                                         #
    #    build_datasets 返回训练集和测试集三元组：                           #
    #    train_list = [train_ref, train_lr, train_hr]                      #
    #    test_list  = [test_ref,  test_lr,  test_hr ]                      #
    #                                                                      #
    #    以 PaviaU 为例（scale_ratio=4, n_select_bands=5, image_size=128）: #
    #      train_ref [1, 103, 608, 340] 训练 GT（中心已挖空）               #
    #      train_lr  [1, 103, 152,  85] 训练低分辨率全波段                  #
    #      train_hr  [1,   5, 608, 340] 训练高分辨率稀疏波段                #
    #      test_ref  [1, 103, 128, 128] 测试 GT（中心 128×128 区域）        #
    #      test_lr   [1, 103,  32,  32] 测试低分辨率全波段                  #
    #      test_hr   [1,   5, 128, 128] 测试高分辨率稀疏波段                #
    # ------------------------------------------------------------------ #
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)

    # 根据数据集名称设置对应的波段数量（各高光谱数据集波段数固定）
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    elif args.dataset == 'KSC':
      args.n_bands = 176
    elif args.dataset == 'Urban':
      args.n_bands = 162
    elif args.dataset == 'IndianP':
      args.n_bands = 200
    elif args.dataset == 'Washington':
      args.n_bands = 191

    # ------------------------------------------------------------------ #
    # 2. 初始化日志记录器                                                   #
    #    TensorBoard：每 100 epoch 写入一次指标曲线                         #
    #    txt 文件：每个 epoch 追加写入 RMSE/PSNR/ERGAS/SAM                  #
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'./runs/{args.dataset}_{args.arch}_{timestamp}')
    log_file = f'{args.dataset}_{args.arch}_metrics_{timestamp}.txt'

    # ------------------------------------------------------------------ #
    # 3. 构建模型并移到 GPU                                                 #
    #    SSRNET / SpatRNET / SpecRNET 均使用 SSRNET 类，通过 arch 参数区分  #
    #    三者网络结构的差异在 SSRNET.forward() 中通过 self.arch 分支控制     #
    # ------------------------------------------------------------------ #
    if args.arch == 'SSFCNN':
      model = SSFCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'ConSSFCNN':
      model = ConSSFCNN(args.scale_ratio, 
                        args.n_select_bands, 
                        args.n_bands).cuda()
    elif args.arch == 'TFNet':
      model = TFNet(args.scale_ratio, 
                    args.n_select_bands, 
                    args.n_bands).cuda()
    elif args.arch == 'ResTFNet':
      model = ResTFNet(args.scale_ratio, 
                       args.n_select_bands, 
                       args.n_bands).cuda()
    elif args.arch == 'MSDCNN':
      model = MSDCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
      # SSRNET（完整版）、SpatRNET（仅空间残差）、SpecRNET（仅光谱残差）共用同一类
      model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'SpatCNN':
      model = SpatCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'SpecCNN':
      model = SpecCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()

    # ------------------------------------------------------------------ #
    # 4. 损失函数 & 优化器                                                  #
    #    criterion: MSELoss（像素级均方误差）                               #
    #    optimizer: Adam，学习率 1e-4                                      #
    # ------------------------------------------------------------------ #
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------ #
    # 5. 加载预训练权重（若存在）                                            #
    #    model_path 格式: ./checkpoints/PaviaU_SSRNET.pkl                  #
    #    strict=False：允许参数名不完全匹配（用于跨版本加载）                 #
    # ------------------------------------------------------------------ #
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        # 加载预训练权重后立即评估一次，查看当前模型性能
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                0,
                                args.n_epochs,
                                writer=writer,
                                dataset=args.arch,
                                log_file=log_file)
        print ('psnr: ', recent_psnr)

    # 训练前评估初始性能，作为 best_psnr 基准
    best_psnr = 0
    best_psnr = validate(test_list,
                          args.arch, 
                          model,
                          0,
                          args.n_epochs,
                          writer=writer,
                          dataset=args.arch,
                          log_file=log_file)
    print ('psnr: ', best_psnr)

    # ------------------------------------------------------------------ #
    # 6. 训练主循环                                                         #
    #    每个 epoch:                                                        #
    #      a. train(): 随机裁 patch → 前向传播 → 计算损失 → 反向传播        #
    #      b. validate(): 在测试集上计算 PSNR 等指标                        #
    #      c. 若当前 PSNR 超过历史最优，保存模型权重到 checkpoint            #
    # ------------------------------------------------------------------ #
    print ('Start Training: ')
    for epoch in range(args.n_epochs):
        # 单 epoch 训练：随机裁 patch，前向+反向传播，更新参数
        print ('Train_Epoch_{}: '.format(epoch))
        train(train_list, 
              args.image_size,
              args.scale_ratio,
              args.n_bands, 
              args.arch,
              model, 
              optimizer, 
              criterion, 
              epoch, 
              args.n_epochs)

        # 单 epoch 验证：在测试集上推理，计算图像质量指标
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs,
                                writer=writer,
                                dataset=args.arch,
                                log_file=log_file)
        print ('psnr: ', recent_psnr)

        # 若当前 epoch 的 PSNR 超过历史最优，则保存模型权重
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
          torch.save(model.state_dict(), model_path)
          print ('Saved!')
          print ('')

    print ('best_psnr: ', best_psnr)
    
    # 关闭 TensorBoard 写入器，确保数据刷盘
    writer.close()

if __name__ == '__main__':
    main()
