import torch.nn as nn 
import torch
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.nn.functional as F


class HeartNet(nn.Module):
    def __init__(self, num_classes=7):
        # Input x is (128, 128, 1
        super(HeartNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16*16*256, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(2048, eps=0.001),
            nn.Linear(2048, num_classes)
            )

        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.xavier_uniform_(self.classifier[4].weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 16 * 256)
        x = self.classifier(x)
        return x

# Testing
model = HeartNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
