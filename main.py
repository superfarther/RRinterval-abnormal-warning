import random
import os.path as osp
import scipy.io as sio
import numpy as np
import os
import matplotlib
from torch.optim.lr_scheduler import MultiStepLR
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from torchsummary import summary
from torchvision import transforms as T
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

"""控制随机数种子"""
def seed_torch(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_torch()

from models import *
from datasets import *


"""—————— 函数区 ————————"""
"""—————— 计算正确率函数 ——————"""
def rightness(predictions, labels):
    # 对于任意一行（一个样本）的输出值的第一个维度，求最大，得到每一行的最大元素的下标
    pred = torch.max(predictions.data, dim=1)[1]
    # 将下标与labels中包含的类别进行比较，并累积得到比较正确的数量
    rights = pred.eq(labels.data.view_as(pred)).sum()
    # 返回正确的数量和这一次一共比较了多少元素
    return rights, len(labels)

"""—————— 训练模型函数 ——————"""
def train_model(data, labels, cur_epoch):
    # 给网络模型做标记，标志说模型正在训练集上训练
    # 这种区分主要是为了打开 net 的 training 标志
    # 从而决定是否运行 dropout 与 batchNorm
    model.train()
    output = model(data)        # 神经网络完成一次前馈的计算过程，得到预测输出output
    loss_CE = criterion_CE(output, labels)    # 将output与标签target比较，计算误差
    optimizer_CE.zero_grad()  # 清空梯度
    loss_CE.backward()  # 反向传播
    optimizer_CE.step()  # 一步随机梯度下降算法
    right = rightness(output, labels)  # 计算准确率所需数值，返回数值为（正确样例数，样本总数）
    return right, loss_CE

"""—————— 校验模型函数 ——————"""
def evaluation_model():
    # net.eval() 给网络模型做标记，标志说模型现在是验证模式
    # 此方法将模型 net 的 training 标志设置为 False
    # 模型中将不会运行 dropout 与 batchNorm
    model.eval()

    # 记录校验数据集准确率的容器
    val_rights = []


    """开始在校验数据集上做循环，计算校验集上面的准确率"""
    for (vibration, labels) in val_loader:
        vibration = vibration.permute(0, 2, 1)
        vibration = vibration.type(torch.cuda.FloatTensor)
        labels = labels.to(torch.int64)
        labels = labels.to(device)

        output = model(vibration)

        # 统计正确数据次数，得到：（正确样例数，batch总样本数）
        right = rightness(output, labels)

        # 加入到容器中，以共后面计算正确率使用
        val_rights.append(right)

    return val_rights


"""—————— 生成实例 ——————"""
num_classes = 2    # 故障类型数量

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())

model = Alexnet(n_class=num_classes)
model.to(device)

# 配置超参数
learning_rate_CE = 0.001
criterion_CE = nn.CrossEntropyLoss()       # 损失函数为交叉熵损失函数
optimizer_CE = torch.optim.Adam(model.parameters(), lr=learning_rate_CE) # 训练算法为 Adam
scheduler_CE = MultiStepLR(optimizer_CE, milestones=[10,20,30,40], gamma=0.1)
num_epoches = 35


"""—————— 正式训练 ——————"""
total_train_step = 1
for epoch in range(1, num_epoches+1):
    print("--------第 {} 轮训练开始--------".format(epoch))
    train_rights = []   # 记录训练数据集准确率的容器

    # 训练开始
    model.train()
    for batch_idx, (vibration, labels) in enumerate(train_loader):  # 取出一个可迭代对象的内容
        vibration = vibration.permute(0, 2, 1)
        vibration = vibration.type(torch.cuda.FloatTensor)
        vibration = vibration.to(device)
        labels = labels.to(torch.int64)
        labels = labels.to(device)

        # 调用模型训练函数
        right, loss_CE = train_model(vibration, labels, epoch)
        # 将计算结果装到列表容器train_rights中
        train_rights.append(right)

        if batch_idx % 100 == 0 and batch_idx != 0:
            print("训练次数: {}, \t交叉熵Loss: {:.6f}".format(total_train_step, loss_CE.item()))
            total_train_step += 1

    scheduler_CE.step()
    with torch.no_grad():
        # 调用模型验证函数
        val_rights = evaluation_model()

        # 统计验证模型时的正确率
        """ val_r 为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数 """
        val_r = (sum([tup[0] for tup in val_rights]), sum(tup[1] for tup in val_rights))

        # 统计上面训练模型时的正确率
        """ train_r 为一个二元组，分别记录目前已经经历过所有训练集集中分类正确的数量和该集合中总的样本数 """
        train_r = (sum([tup[0] for tup in train_rights]), sum(tup[1] for tup in train_rights))

        # 计算并打印出模型在训练时和在验证时的准确率
        # train_r[0]/train_r[1]就是训练集的分类准确度，同样，val_r[0]/val_r[1]就是校验集上的分类准确度
        print("训练周期:{} [{}/{} ({:.0f}%)]\t训练正确率:{:.2f}%\t校验正确率:{:.2f}%".format(
            epoch, batch_idx * batch_size, len(train_dataset), 100. * batch_idx / len(train_loader)
            , 100. * train_r[0].cpu().numpy() / train_r[1], 100. * val_r[0].cpu().numpy() / val_r[1])
        )
print("训练结束")

torch.save(model, 'model_yichang.pth')