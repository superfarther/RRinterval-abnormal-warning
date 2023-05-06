import torch
from torch import nn
from torchsummary import summary

""" —————— AlexNet网络 batch*1*1000 ==> batch*10 ——————"""
class Alexnet(nn.Module):
    def __init__(self, in_dim=1, n_class=4):
        super(Alexnet, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module("conv1", nn.Conv1d(in_channels=in_dim, out_channels=48, kernel_size=11, stride=4,
                                             padding=0))
        # output = [48, 248, 1]
        layer1.add_module("bn1", nn.BatchNorm1d(48))
        layer1.add_module("relu1", nn.ReLU(inplace=True))
        layer1.add_module("pool1", nn.AvgPool1d(kernel_size=3, stride=2))
        # output = [48, 123, 1]
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module("conv2", nn.Conv1d(in_channels=48, out_channels=128, kernel_size=5, stride=1,
                                             padding=2))
        # output = [128, 123, 1]
        layer2.add_module("bn2", nn.BatchNorm1d(128))
        layer2.add_module("relu2", nn.ReLU(inplace=True))
        layer2.add_module("pool2", nn.AvgPool1d(kernel_size=3, stride=2))
        # output = [128, 61, 1]
        self.layer2 = layer2

        layer6 = nn.Sequential()
        layer6.add_module("fc1", nn.Linear(in_features=7936, out_features=1024))
        layer6.add_module("relu3", nn.ReLU(inplace=True))
        layer6.add_module("drop1", nn.Dropout(p=0.5))
        layer6.add_module("fc2", nn.Linear(in_features=1024, out_features=10))
        layer6.add_module("relu7", nn.ReLU(inplace=True))
        layer6.add_module("fc3", nn.Linear(in_features=10, out_features=2))

        self.layer6 = layer6


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)   # 将 (batch, 256, 6, 6) 展平为 (batch, 256*6*6)
        output = self.layer6(x)
        return output

if __name__ == '__main__':
    """—————— 生成实例 ——————"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.is_available())

    model = Alexnet()
    model.to(device)
    summary(model, (1, 1024))
