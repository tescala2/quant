import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.utils.loops import run


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.name = 'cnn'

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding=4)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=9, padding=4)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=9, padding=4)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(8)

        self.lin1 = nn.Linear(2400, 16)
        self.lin2 = nn.Linear(16, 8)
        self.lin3 = nn.Linear(8, 1)

    def forward(self, x):
        # print(x.shape)
        conv_res = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        conv_res2 = F.relu(self.bn2(self.conv2(conv_res)))
        # print(x.shape)
        conv_res3 = F.relu(self.bn3(self.conv3(conv_res2)))
        # print(x.shape)
        x = F.relu(self.bn4(self.conv4(conv_res3)) + conv_res2)
        # print(x.shape)
        x = F.relu(self.bn5(self.conv5(x)) + conv_res)
        # print(x.shape)
        x = x.view(-1, x.shape[1] * x.shape[2])
        # print(x.shape)
        x = F.relu(self.lin1(x))
        # print(x.shape)
        x = F.relu(self.lin2(x))

        x = self.lin3(x)

        return x.reshape(-1)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN()

    run(model, name='cnn', device=device, epochs=10, feature_first=True)
