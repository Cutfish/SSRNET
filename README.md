# SSRNET

SSRNET for hyperspectral and multispectral image fusion.
This is the code of X. Zhang, W. Huang, Q. Wang, and X. Li, “SSR-NET: Spatial-Spectral Reconstruction Network for Hyperspectral and Multispectral Image Fusion,”  IEEE Transactions on Geoscience and Remote Sensing (T-GRS), 2020.

# 运行指令

```shell
# 运行项目
conda run -n ssrnet python main.py -dataset PaviaU -arch SSRNET --n_epochs 1000
# 打开tensorboard监控
conda run -n ssrnet tensorboard --logdir=./runs --port=6006
```
