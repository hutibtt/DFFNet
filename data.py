# import os
#
# # def generate_train_txt_with_absolute_path(root, output_file='train.txt', prefix='Tomato__'):
# #     """
# #     修复类别加载问题，并生成包含绝对路径的 train.txt 文件。
# #
# #     Args:
# #         root (str): 训练数据集的根目录，每个类别应在单独的子目录中。
# #         output_file (str): 输出的文件名。
# #         prefix (str): 文件夹名前缀，用于提取类别名称。
# #     """
# #     # 获取所有类别文件夹
# #     classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
# #
# #     # 创建类别名称到索引的映射
# #     class_to_idx = {cls[len(prefix):]: idx for idx, cls in enumerate(classes)}
# #     print(f"检测到的类别及索引映射：{class_to_idx}")
# #
# #     # 写入 train.txt 文件
# #     with open(output_file, 'w') as f:
# #         for cls in classes:
# #             cls_name = cls[len(prefix):]  # 提取去掉前缀后的类别名
# #             idx = class_to_idx.get(cls_name, None)
# #
# #             # 检查类别是否解析正确
# #             if idx is None:
# #                 print(f"警告：未能解析类别 {cls}，跳过。")
# #                 continue
# #
# #             cls_dir = os.path.join(root, cls)
# #             print(f"处理类别：{cls}，索引：{idx}")
# #
# #             # 遍历该类别下的所有图像文件
# #             found_images = False
# #             for image_name in os.listdir(cls_dir):
# #                 if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # 确保是图像文件
# #                     found_images = True
# #                     absolute_image_path = os.path.abspath(os.path.join(cls_dir, image_name))  # 转换为绝对路径
# #                     f.write(f"{absolute_image_path} {idx}\n")
# #
# #             if not found_images:
# #                 print(f"警告：类别目录 {cls_dir} 下未找到图像文件！")
# #
# #     print(f"train.txt 已生成，保存路径：{output_file}")
# #
# # # 使用示例
# # root_dir = '/mnt/sdb1/data/pv/train'  # 数据集根目录
# # output_file = '/mnt/sdb1/data/pv/train.txt'
# # generate_train_txt_with_absolute_path(root_dir, output_file, prefix='Tomato__')
# # # 工具类
#
# # import os
# #
# # # 指定 test 文件夹的路径
# # root_folder = "/mnt/sdb1/data/pv/train"  # 替换为实际路径
# #
# # # 遍历 test 文件夹及其所有子文件夹
# # for subdir, _, files in os.walk(root_folder):
# #     for file in files:
# #         file_path = os.path.join(subdir, file)
# #         # 如果文件名包含空格，则删除空格
# #         if " " in file:
# #             new_name = file.replace(" ", "")
# #             new_path = os.path.join(subdir, new_name)
# #             os.rename(file_path, new_path)
# #             print(f"重命名: {file} -> {new_name}")
# # import os
# #
# # # 指定 train 文件夹的路径
# # train_path = "/mnt/sdb1/data/pv/test"  # 替换为实际路径
# #
# # # 遍历 train 文件夹下的所有文件夹
# # for item in os.listdir(train_path):
# #     item_path = os.path.join(train_path, item)
# #     if os.path.isdir(item_path):  # 判断是否为文件夹
# #         # 如果文件夹名包含空格，将空格替换为下划线
# #         if " " in item:
# #             new_name = item.replace(" ", "_")
# #             new_path = os.path.join(train_path, new_name)
# #             os.rename(item_path, new_path)
# #             print(f"重命名: {item} -> {new_name}")
#
# 工具类
# import os
# import random
# import shutil
# from shutil import copy2
#
#
# def data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0, test_scale=0.3):
#     '''
#     读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
#     :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
#     :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
#     :param train_scale: 训练集比例
#     :param val_scale: 验证集比例
#     :param test_scale: 测试集比例
#     :return:
#     '''
#     print("开始数据集划分")
#     class_names = os.listdir(src_data_folder)
#     # 在目标目录下创建文件夹
#     split_names = ['train', 'val', 'test']
#     for split_name in split_names:
#         split_path = os.path.join(target_data_folder, split_name)
#         if os.path.isdir(split_path):
#             pass
#         else:
#             os.mkdir(split_path)
#         # 然后在split_path的目录下创建类别文件夹
#         for class_name in class_names:
#             class_split_path = os.path.join(split_path, class_name)
#             if os.path.isdir(class_split_path):
#                 pass
#             else:
#                 os.mkdir(class_split_path)
#
#     # 按照比例划分数据集，并进行数据图片的复制
#     # 首先进行分类遍历
#     for class_name in class_names:
#         current_class_data_path = os.path.join(src_data_folder, class_name)
#         current_all_data = os.listdir(current_class_data_path)
#         current_data_length = len(current_all_data)
#         current_data_index_list = list(range(current_data_length))
#         random.shuffle(current_data_index_list)
#
#         train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
#         val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
#         test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
#         train_stop_flag = current_data_length * train_scale
#         val_stop_flag = current_data_length * (train_scale + val_scale)
#         current_idx = 0
#         train_num = 0
#         val_num = 0
#         test_num = 0
#         for i in current_data_index_list:
#             src_img_path = os.path.join(current_class_data_path, current_all_data[i])
#             if current_idx <= train_stop_flag:
#                 copy2(src_img_path, train_folder)
#                 # print("{}复制到了{}".format(src_img_path, train_folder))
#                 train_num = train_num + 1
#             elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
#                 copy2(src_img_path, val_folder)
#                 # print("{}复制到了{}".format(src_img_path, val_folder))
#                 val_num = val_num + 1
#             else:
#                 copy2(src_img_path, test_folder)
#                 # print("{}复制到了{}".format(src_img_path, test_folder))
#                 test_num = test_num + 1
#
#             current_idx = current_idx + 1
#
#         print("*********************************{}*************************************".format(class_name))
#         print(
#             "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
#         print("训练集{}：{}张".format(train_folder, train_num))
#         print("验证集{}：{}张".format(val_folder, val_num))
#         print("测试集{}：{}张".format(test_folder, test_num))
#
#
# if __name__ == '__main__':
#     src_data_folder = r"/mnt/sdb1/data/plant-pathology-2021-fgvc8/sorted_train_images"
#     target_data_folder = r"/mnt/sdb1/data/plant-pathology-2021-fgvc8/dataset"
#     data_set_split(src_data_folder, target_data_folder)

# import os
# from PIL import Image, ImageStat
#
# # 输入文件夹路径
# input_dir = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/sorted_train_images/healthy"
#
# # 曝光过度的亮度阈值（可根据需要调整）
# brightness_threshold = 180  # 亮度值范围是 0-255，值越高越亮
#
# # 统计曝光过度的图片数量和文件名
# overexposed_images = []
#
# # 遍历文件夹中的所有图像
# for image_name in os.listdir(input_dir):
#     image_path = os.path.join(input_dir, image_name)
#     if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
#         continue  # 跳过非图像文件
#
#     try:
#         # 打开图像并转换为灰度图
#         image = Image.open(image_path).convert("L")  # 转换为灰度图
#         stat = ImageStat.Stat(image)
#         brightness = stat.mean[0]  # 计算平均亮度
#
#         # 如果亮度超过阈值，记录图像信息
#         if brightness > brightness_threshold:
#             overexposed_images.append((image_name, brightness))
#
#     except Exception as e:
#         print(f"处理文件时出错: {image_path}, 错误: {e}")
#
# # 打印统计结果
# print(f"检测到 {len(overexposed_images)} 张曝光过度的图片：")
# for image_name, brightness in overexposed_images:
#     print(f"  - {image_name} (亮度: {brightness:.2f})")

# import os
#
# # 输入训练文件夹路径和输出文件路径
# train_folder = '/mnt/sdb1/data/plant-pathology-2021-fgvc8/dataset/test'  # 训练图像所在文件夹
# output_file = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/dataset/test.txt"
#
# # 获取类别文件夹（按字母顺序排序）
# classes = sorted(os.listdir(train_folder))
# class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}
#
# # 遍历所有类别文件夹，生成文件路径和标签
# with open(output_file, "w") as f:
#     for cls_name, cls_index in class_to_index.items():
#         cls_folder = os.path.join(train_folder, cls_name)
#         if not os.path.isdir(cls_folder):
#             continue  # 跳过非文件夹
#
#         for image_name in os.listdir(cls_folder):
#             if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
#                 image_path = os.path.join(cls_folder, image_name)
#                 f.write(f"{image_path} {cls_index}\n")
#
# print(f"`train.txt` 文件已生成，保存路径为: {output_file}")
#
# import os
#
# # 输入路径
# output_dir = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/sorted_train_images/complex"
#
# # 删除以 "_aug_" 开头的文件
# for image_name in os.listdir(output_dir):
#     if "_aug_" in image_name:  # 检查文件名中是否包含 "_aug_"
#         file_path = os.path.join(output_dir, image_name)
#         os.remove(file_path)  # 删除文件
#         print(f"已删除: {file_path}")
#
# print(f"清理完成，已删除所有包含 '_aug_' 的增强图片。")

# import os
# import pandas as pd
# import shutil
#
# # 定义需要删除的类别
# categories_to_remove = [
#     "scab frog_eye_leaf_spot complex",
#     "scab frog_eye_leaf_spot",
#     "frog_eye_leaf_spot complex",
#     "rust frog_eye_leaf_spot",
#     "powdery_mildew complex",
#     "complex",
#     "rust complex"
# ]
#
# # 加载 train.csv
# train_csv_path = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/train.csv"
# train_images_dir = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/train_images"
# output_dir = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/filter_train_images"  # 保存过滤后图片的目录
#
# df = pd.read_csv(train_csv_path)
#
# # 筛选出需要保留的行
# filtered_df = df[~df['labels'].isin(categories_to_remove)]
#
# # 创建输出目录
# os.makedirs(output_dir, exist_ok=True)
#
# # 移动保留的图片到新的目录
# for _, row in filtered_df.iterrows():
#     image_path = os.path.join(train_images_dir, row['image'])
#     if os.path.exists(image_path):
#         shutil.copy(image_path, output_dir)
#
# # 保存过滤后的CSV
# filtered_csv_path = os.path.join(output_dir, "filtered_train.csv")
# filtered_df.to_csv(filtered_csv_path, index=False)
#
# print(f"完成过滤，过滤后的图片保存在: {output_dir}")
# print(f"过滤后的CSV文件保存为: {filtered_csv_path}")
# import os
# import torch
# import random
# import numpy as np
# from PIL import Image
# from torchvision import transforms
#
# # 输入和输出路径
# source_dir = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/sorted_train_images/rust"
# output_dir = "/mnt/sdb1/data/plant-pathology-2021-fgvc8/sorted_train_images/rust"
# os.makedirs(output_dir, exist_ok=True)
#
# # 设置设备为 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # 数据增强操作
# def augment_image(image_tensor):
#     augmented_images = []
#
#     # 1. 随机旋转（仅旋转 180°）
#     rotated = transforms.functional.rotate(image_tensor, 180)
#     augmented_images.append(rotated)
#
#     # 2. 镜像
#     mirrored = transforms.functional.hflip(image_tensor)
#     augmented_images.append(mirrored)
#
#     # 3. 添加高斯噪声
#     def add_gaussian_noise(tensor):
#         noise = torch.randn_like(tensor) * 0.06  # 标准差控制噪声强度
#         noisy_tensor = torch.clamp(tensor + noise, 0, 1)  # 保证值范围在 [0, 1]
#         return noisy_tensor
#
#     noisy = add_gaussian_noise(image_tensor)
#     augmented_images.append(noisy)
#
#     # 4. 应用颜色抖动
#     color_jitter = transforms.ColorJitter(
#         brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
#     )
#     jittered = color_jitter(image_tensor)
#     augmented_images.append(jittered)
#
#     return augmented_images
#
# # 定义图像转换为张量及恢复为图像的操作
# to_tensor = transforms.ToTensor()
# to_image = transforms.ToPILImage()
#
# # 处理每张图像
# for image_name in os.listdir(source_dir):
#     image_path = os.path.join(source_dir, image_name)
#     if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
#         continue
#
#     # 加载图像并转换为张量
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = to_tensor(image).to(device)
#
#     # 数据增强
#     augmented_images = augment_image(image_tensor)
#
#     # 保存增强后的图像
#     for idx, aug_tensor in enumerate(augmented_images):
#         aug_image = to_image(aug_tensor.cpu())
#         new_name = f"{os.path.splitext(image_name)[0]}_aug_{idx}.jpg"
#         aug_image.save(os.path.join(output_dir, new_name))
#
# print(f"数据增强完成，增强后的图像保存在: {output_dir}")
import os
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

# 输入和输出路径
source_dir = "/home/pl/htt/swin-frequence/cam"
output_dir = "/home/pl/htt/swin-frequence/cam_results"
os.makedirs(output_dir, exist_ok=True)

# 添加高斯噪声
def add_gaussian_noise(image):
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 15, img_array.shape)  # 均值=0，标准差=15
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# 颜色抖动
color_jitter = transforms.ColorJitter(
    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
)

# 定义固定4种增强操作
def augment_image(image):
    return [
        image.rotate(180),                # 1. 旋转180°
        ImageOps.mirror(image),            # 2. 镜像
        add_gaussian_noise(image),         # 3. 高斯噪声
        color_jitter(image)                 # 4. 颜色抖动
    ]

# 统计图像数量
original_images = os.listdir(source_dir)
original_count = len([f for f in original_images if f.lower().endswith((".png", ".jpg", ".jpeg"))])

# 处理每张图像
for image_name in original_images:
    image_path = os.path.join(source_dir, image_name)
    if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 保存原始图像
    output_path = os.path.join(output_dir, image_name)
    image.save(output_path)

    # 生成4个固定增强版本
    augmented_images = augment_image(image)
    for idx, aug_image in enumerate(augmented_images):
        new_name = f"{os.path.splitext(image_name)[0]}_aug_{idx}.jpg"
        aug_image.save(os.path.join(output_dir, new_name))

print(f"数据增强完成，原始图像数量: {original_count}，扩充后图像数量: {original_count * 5}。增强后的图像保存在: {output_dir}")

# #
# import os
# import shutil
# import pandas as pd
#
# # 定义路径变量
# data_dir = '/mnt/sdb1/data/plant-pathology-2021-fgvc8'
# train_images_dir = os.path.join(data_dir, 'train_images')
# sorted_train_images_dir = os.path.join(data_dir, 'sorted_train_images')
# csv_file = os.path.join(data_dir, 'train.csv')
#
# # 加载CSV文件
# train_df = pd.read_csv(csv_file)
#
# # 创建每个类的文件夹
# for label in train_df['labels'].unique():
#     label_dir = os.path.join(sorted_train_images_dir, str(label))
#     if not os.path.exists(label_dir):
#         os.makedirs(label_dir)
#
# # 将图片按照标签分类
# for index, row in train_df.iterrows():
#     image_id = row['image']
#     label = row['labels']
#
#     # 源图片路径
#     image_path = os.path.join(train_images_dir, image_id)
#
#     # 目标路径（根据标签创建文件夹）
#     target_path = os.path.join(sorted_train_images_dir, str(label), image_id)
#
#     # 移动图片到目标文件夹
#     shutil.move(image_path, target_path)
#

#
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from dataset import YourDataset  # 你需要根据你的数据集定义自己的数据加载类
#
# # 假设你已经有训练好的模型
# model = torch.load('/home/pl/htt/swin-frequence/output/swin_tiny_patch4_window7_224_22k/default/best.pth')  # 加载模型
# model.eval()  # 切换到评估模式
#
# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((448, 448)),  # 这里调整为你输入图像的尺寸
#     transforms.ToTensor(),  # 转为Tensor
# ])
#
# # 加载测试集
# test_dataset = YourDataset(root_dir='/mnt/sdb1/data/plant-pathology-2021-fgvc8/data', transform=transform, split='test')  # 自定义数据集
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
#
# # 初始化标签和预测列表
# true_labels = []
# predictions = []
#
# # 遍历测试集
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         # 将输入数据转移到设备上（如果使用GPU）
#         inputs = inputs.cuda() if torch.cuda.is_available() else inputs
#         labels = labels.cuda() if torch.cuda.is_available() else labels
#
#         # 进行预测
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#
#         # 记录预测和真实标签
#         true_labels.extend(labels.cpu().numpy())
#         predictions.extend(predicted.cpu().numpy())
#
# # 计算混淆矩阵
# cm = confusion_matrix(true_labels, predictions)
#
# # 绘制混淆矩阵
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix.png')  # 保存为PNG文件
# plt.close()
#
