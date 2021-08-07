import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataset_generation import data_generator
from model import ArcFace

CLASS_NUM = 12
EPOCHS = 30
BATCH_SIZE = 16

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

datagen = data_generator(DATASET_PATH = 'C:/projects/dataset/thermal_image/80_60_train_dataset/', shuffle_sel=True)
arcface = ArcFace(class_num=CLASS_NUM)
optimizer = optim.Adam(arcface.parameters())
# loss_cal = nn.CrossEntropyLoss()
loss_cal = LabelSmoothLoss()
data_class_set, data_array, label, data_name = datagen.data_label_set_gen()


for epoch in range(EPOCHS):
    for step in range(int(len(data_array)/BATCH_SIZE)):
        indices = np.random.choice(len(data_array), BATCH_SIZE)
        batch_image = []
        batch_label = []
        for index in indices:
            sample = data_array[index]
            sample = np.transpose(sample)
            batch_image.append(sample)
            batch_label.append(label[index])

        prob_feature = arcface.forward(torch.tensor(batch_image), torch.tensor(batch_label))
        loss = loss_cal(prob_feature, torch.LongTensor(batch_label))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('steps:',step,'/',int(len(data_array)/BATCH_SIZE),'epoch:', epoch,'/',EPOCHS,'loss :',round(float(loss.detach()),4))
    torch.save(arcface, 'arcface_model_'+str(epoch)+'.pt')