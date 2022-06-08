import torch
import torch.nn as nn
import torch.nn.functional as F
from source.model.coordconv import CoordConv2d

#nclasses = 17 #KETIDB+TL SEOUL --> 5 classes // 1st release
class TrafficLightNet_128x128_LSTM(nn.Module):
    def __init__(self, nfeature=14700, nhidden=500, nseq=10, nlayers=20, nclasses=7):
        super(TrafficLightNet_128x128_LSTM, self).__init__()

        self.nseq = nseq
        self.conv1 = CoordConv2d(in_channels = 3, out_channels = 64, kernel_size = 3)
        self.conv1_bn = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = CoordConv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv2_bn = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 250, kernel_size = 1)
        self.conv3_bn = nn.BatchNorm2d(num_features=250)

        self.conv4 = nn.Conv2d(in_channels = 250, out_channels = 300, kernel_size = 1)
        self.conv4_bn = nn.BatchNorm2d(num_features=300)
        self.dropout = nn.Dropout(p = 0.1)

        self.lstm = nn.LSTM(
            input_size = nfeature, hidden_size=nhidden, batch_first=True)
        
        self.linear = nn.Linear(in_features=nhidden, out_features=nclasses)

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(-1, *x.shape[2:])
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.conv1_bn(x)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout(self.conv2_bn(x))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.dropout(self.conv3_bn(x))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = self.conv4_bn(x)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2_bn(self.dropout(self.conv2(x))))
        x = self.pool(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.conv4_bn(self.dropout(self.conv4(x))))
        x = self.pool(x)

        x = x.reshape(-1, 300 * 7 * 7)
        x = x.reshape(batch, self.nseq, 300 * 7 * 7)
        lstm_out, (ht, ct) = self.lstm(x)
        y = self.linear(ht[-1])
        return y


class TrafficLightNet_64x32_LSTM(nn.Module):
    def __init__(self, nfeature=900, nhidden=500, nseq=10, nlayers=20, nclasses=7):
        super(TrafficLightNet_64x32_LSTM, self).__init__()

        self.nseq = nseq
        self.conv1 = CoordConv2d(in_channels = 3, out_channels = 64, kernel_size = 3)
        self.conv1_bn = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = CoordConv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv2_bn = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 250, kernel_size = 1)
        self.conv3_bn = nn.BatchNorm2d(num_features=250)

        self.conv4 = nn.Conv2d(in_channels = 250, out_channels = 300, kernel_size = 1)
        self.conv4_bn = nn.BatchNorm2d(num_features=300)
        self.dropout = nn.Dropout(p = 0.1)

        self.lstm = nn.LSTM(
            input_size = nfeature, hidden_size=nhidden, batch_first=True)
        
        self.linear = nn.Linear(in_features=nhidden, out_features=nclasses)

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(-1, *x.shape[2:])
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.conv1_bn(x)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout(self.conv2_bn(x))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.dropout(self.conv3_bn(x))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = self.conv4_bn(x)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2_bn(self.dropout(self.conv2(x))))
        x = self.pool(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.conv4_bn(self.dropout(self.conv4(x))))
        x = self.pool(x)

        x = x.reshape(-1, 300 * 3 * 1)
        x = x.reshape(batch, self.nseq, 300 * 3 * 1)
        lstm_out, (ht, ct) = self.lstm(x)
        y = self.linear(ht[-1])
        return y
