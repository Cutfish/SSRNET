from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils import *
import cv2
import pdb
from datetime import datetime
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam


def validate(test_list, arch, model, epoch, n_epochs, writer=None, dataset='dataset', log_file=None):
    test_ref, test_lr, test_hr = test_list
    model.eval()

    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = to_var(test_ref).detach()
        lr = to_var(test_lr).detach()
        hr = to_var(test_hr).detach()
        if arch == 'SSRNet':
            out, _, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpat':
            _, out, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpec':
            _, _, out, _, _, _ = model(lr, hr)
        else:
            out, _, _, _, _, _ = model(lr, hr)

        ref = ref.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out)
        sam = calc_sam(ref, out)

        # 保存 txt 文件（每个 epoch）
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{dataset}_metrics_{timestamp}.txt'
        
        with open(log_file, 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) + ',' + '\n')
        
        # 只在每 100 个 epoch 记录到 TensorBoard
        if epoch % 100 == 0:
            if writer:
                writer.add_scalar(f'metrics/{dataset}/RMSE', rmse, epoch)
                writer.add_scalar(f'metrics/{dataset}/PSNR', psnr, epoch)
                writer.add_scalar(f'metrics/{dataset}/ERGAS', ergas, epoch)
                writer.add_scalar(f'metrics/{dataset}/SAM', sam, epoch)

    return psnr