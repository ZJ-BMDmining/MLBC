import torch.nn as nn
import torch.nn.functional as F
class NeuralNetwork(nn.Module):
    def __init__(self,input,num_classes=4):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        # print(x.shape)
        x = F.leaky_relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return F.softmax(x, dim=1)