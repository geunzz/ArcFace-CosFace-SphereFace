import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class CombineNet(nn.Module):
    def __init__(self, s=128.0, arc_margin=0.5, cos_margin=0.4, sphere_margin=1.4, class_num=20):
        super(CombineNet, self).__init__()
        self.s = s
        self.arc_margin = arc_margin
        self.cos_margin = cos_margin
        self.sphere_margin = sphere_margin

        self.class_num = class_num
        self.conv1 = nn.Conv2d(3, 128, 3, padding=0)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense = nn.Linear(29952, 128)
        self.arc_dense = nn.Linear(128, 64)
        self.cos_dense = nn.Linear(128, 64)
        self.sphere_dense = nn.Linear(128, 64)
        self.arc_weight = Parameter(torch.FloatTensor(class_num, 64))
        self.cos_weight = Parameter(torch.FloatTensor(class_num, 64))
        self.sphere_weight = Parameter(torch.FloatTensor(class_num, 64))

        nn.init.xavier_uniform_(self.arc_weight)
        nn.init.xavier_uniform_(self.cos_weight)
        nn.init.xavier_uniform_(self.sphere_weight)

    def forward(self, input):
        input = (input.float()/255).clone().detach()
        input = input.transpose(1,3)
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = torch.flatten(input, start_dim=1)
        net_feature = self.dense(input)

        return net_feature
        
    def arcface(self, net_feature, label):
        each_feature = self.arc_dense(net_feature)
        cosine = F.linear(F.normalize(each_feature), F.normalize(self.arc_weight))
        a_cosine = torch.acos(cosine)
        margin_add = self.arc_margin*torch.eye(self.class_num)[label]
        radian = a_cosine + margin_add
        arc_feature = self.s*torch.cos(radian)
        
        return arc_feature

    def cosface(self, net_feature, label):
        each_feature = self.cos_dense(net_feature)
        cosine = F.linear(F.normalize(each_feature), F.normalize(self.cos_weight))
        margin_minus = self.cos_margin*torch.eye(self.class_num)[label]
        cos_value = cosine - margin_minus
        cos_feature = self.s*cos_value

        return cos_feature

    def sphereface(self, net_feature, label):
        each_feature = self.sphere_dense(net_feature)
        cosine = F.linear(F.normalize(each_feature), F.normalize(self.sphere_weight))
        a_cosine = torch.acos(cosine)
        margin_mul = self.sphere_margin*torch.eye(self.class_num)[label]
        margin_mul = torch.where(margin_mul == 0., torch.tensor(1.), margin_mul)
        radian = a_cosine*margin_mul
        sphere_feature = self.s*torch.cos(radian)

        return sphere_feature
    
    def test(self, input):
        input = (input.float()/255).clone().detach()
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = torch.flatten(input, start_dim=1)
        feature = self.dense(input)
        arc_dense = self.arc_dense(feature)
        cos_dense = self.cos_dense(feature)
        sphere_dense = self.sphere_dense(feature)
        #arcface
        cosine = F.linear(F.normalize(arc_dense), F.normalize(self.arc_weight))
        a_cosine = torch.acos(cosine)
        arc_feature = self.s*torch.cos(a_cosine)
        #cosface
        cosine = F.linear(F.normalize(cos_dense), F.normalize(self.cos_weight))
        cos_feature = self.s*cosine
        #sphereface
        cosine = F.linear(F.normalize(sphere_dense), F.normalize(self.sphere_weight))
        a_cosine = torch.acos(cosine)
        sphere_feature = self.s*torch.acos(a_cosine)
        
        return arc_feature, cos_feature, sphere_feature