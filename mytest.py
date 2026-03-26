from data_loader import build_datasets

# 参数配置
root = './data'
dataset = 'PaviaU'
image_size = 128
n_select_bands = 5
scale_ratio = 4

# 加载数据集
train_list, test_list = build_datasets(root, 
                                       dataset, 
                                       image_size, 
                                       n_select_bands, 
                                       scale_ratio)

# 解包数据
train_ref, train_lr, train_hr = train_list
test_ref, test_lr, test_hr = test_list

# 打印数据集形状
print("=" * 60)
print("训练集 (Training Set):")
print("-" * 60)
print(f"train_ref (全波段参考图): {train_ref.shape}")
print(f"train_lr  (低分辨率):     {train_lr.shape}")
print(f"train_hr  (选定波段):     {train_hr.shape}")

print("\n" + "=" * 60)
print("测试集 (Test Set):")
print("-" * 60)
print(f"test_ref  (全波段参考图): {test_ref.shape}")
print(f"test_lr   (低分辨率):     {test_lr.shape}")
print(f"test_hr   (选定波段):     {test_hr.shape}")

print("\n" + "=" * 60)
print("数据统计:")
print("-" * 60)
print(f"波段总数: {train_ref.shape[1]}")
print(f"选定波段数: {train_hr.shape[1]}")
print(f"空间下采样比例: {scale_ratio}x")
print(f"训练图像尺寸: {train_ref.shape[2]} x {train_ref.shape[3]}")
print(f"测试图像尺寸: {test_ref.shape[2]} x {test_ref.shape[3]}")

print("\n" + "=" * 60)
print("数据范围:")
print("-" * 60)
print(f"train_ref: [{train_ref.min():.2f}, {train_ref.max():.2f}]")
print(f"test_ref:  [{test_ref.min():.2f}, {test_ref.max():.2f}]")
print("=" * 60)