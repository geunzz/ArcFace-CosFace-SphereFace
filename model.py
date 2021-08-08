import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcFace(nn.Module):
    def __init__(self, s=120.0, margin=0.5, class_num=20):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = margin
        self.class_num = class_num
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=0)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense = nn.Linear(14976, 64)
        self.weight = Parameter(torch.FloatTensor(class_num, 64))

        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, label):
        input = (input.float()/255).clone().detach()
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = torch.flatten(input, start_dim=1)
        feature = self.dense(input)

        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        a_cosine = torch.acos(cosine)
        margin_add = self.m*torch.eye(self.class_num)[label]
        radian = a_cosine + margin_add
        margin_feature = self.s*torch.cos(radian)
        
        return margin_feature

    def test(self, input):
        input = (input.float()/255).clone().detach()
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = torch.flatten(input, start_dim=1)
        feature = self.dense(input)
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        a_cosine = torch.acos(cosine)
        feature = self.s*torch.cos(a_cosine)

        return feature
