import os

import pandas as pd
import scipy.io as sio
import torch
import torch.utils.data
import random
import numpy as np
from torchvision import transforms as T

""" 数据预处理 """
# batch 大小
batch_size = 4
Path_unusual = "Q:\data\DataSets\异常预警\标记为异常的数据"
alldir_list_unusual = os.listdir(Path_unusual)
print(len(alldir_list_unusual))
dir_paths_unusual = []
for path in alldir_list_unusual:
    dir_paths_unusual.append(os.path.join(Path_unusual, path))

Path_normal = "Q:\data\DataSets\异常预警\标注为正常的数据"
alldir_list_normal = os.listdir(Path_normal)
print(len(alldir_list_normal))
dir_paths_normal = []
for path in alldir_list_normal:
    dir_paths_normal.append(os.path.join(Path_normal, path))

Datas_VT = []
Datas_VF = []
Datas_Normal = []
Datas = []
Labels = []
for path in dir_paths_unusual:
    df = pd.read_csv(path, header=None)
    if "vt" in path:
        Datas_VT.append(np.array(df.values))
        Datas.append(np.array(df.values))
        Labels.append(1)
    if "vf" in path:
        Datas_VF.append(np.array(df.values))
        Datas.append(np.array(df.values))
        Labels.append(1)
for path in dir_paths_normal:
    try:
        df = pd.read_csv(path,header=None)
        Datas_Normal.append(np.array(df.values)[0:1024])
        Datas.append(np.array(np.array(df.values)[0:1024]))
        Labels.append(0)
    except:
        print(path)
Datas = np.array(Datas)
Datas = Datas.astype(float)

""" 自制数据集 """
class CWRU_Dataset(torch.utils.data.Dataset):
    """ 自制数据集 """
    def __init__(self, datas, labels, mode="train"):
        """
        :param datas: 总的数据 数组
        :param labels: 对应标签
        :param mode:
        """
        self.mode = mode
        data_len = len(datas)
        datas = np.array(datas)
        labels = np.array(labels)

        # 验证集：2 / 10；训练集：8 / 10
        if mode == "train":
            indices = [i for i in range(data_len) if i % 10 != 0]
        elif mode == "valid":
            indices = [i for i in range(data_len) if i % 10 == 0]
        self.datas = datas[indices]
        self.labels = labels[indices]
        result = pd.value_counts(self.labels)
        print(result)
        self.real_len = len(self.datas)
        print(datas.dtype)
        print(labels.dtype)


    def __getitem__(self, index):
        if self.mode == 'train':
            transform = T.ToTensor()
        else:
            transform = T.ToTensor()
        datas = transform(self.datas)
        return self.datas[index], self.labels[index]
    def __len__(self):
        return self.real_len

train_dataset = CWRU_Dataset(Datas, Labels, mode="train")
val_dataset = CWRU_Dataset(Datas, Labels, mode="valid")

train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
print("训练数据集的长度为:{}".format(train_data_size))
print("校验数据集的长度为:{}".format(val_data_size))

# 训练数据集的加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


# 校验数据集的加载器
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

