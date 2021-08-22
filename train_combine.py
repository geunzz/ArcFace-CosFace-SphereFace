import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
from collections import deque

CLASS_NUM = 12
EPOCHS = 20
BATCH_SIZE = 16
REPLAY_MEMORY = 20000
BETA = 0.95
USE_PER = False

class ReplayMemory:
    def __init__(self, model, loss_cal, use_per=True):
        self.buffer = deque(maxlen=REPLAY_MEMORY)
        self.use_per = use_per
        self.beta = BETA
        self.model = model
        self.loss_cal = loss_cal

    def write(self, data_array, label):
        for i in range(len(data_array)):
            self.buffer.append([1.0, 0 ,data_array[i], label[i]])

    def sample(self, epoch):
        feature, label = [], []
        if self.use_per and epoch != 0:
            #PER 적용 시 우선순위에 따라 정렬 후 sampling
            prob_list = []            
            sum_error = 0.000000001
            for i in range(len(self.buffer)):
                sum_error = sum_error + float(self.buffer[i][0])
            prob_list = list(np.array(self.buffer, dtype=object)[:,0])
            prob_list = np.array(prob_list)/float(sum_error)
            indices = np.random.choice(len(self.buffer), BATCH_SIZE, p=prob_list)
        
        else:
            indices = random.sample(range(0,len(self.buffer)),BATCH_SIZE)
        
        for index in indices:
            feature.append(self.buffer[index][2]) #feature
            label.append(self.buffer[index][3]) #label

        if epoch != 0:
            #sampling 된 이력이 있으면 counter 올려주고 TD error를 annealing 시킴
            self.buffer[index][1] = self.buffer[index][1] + 1
            self.buffer[index][0] = self.buffer[index][0]*self.beta

        return torch.tensor(feature), torch.tensor(label)

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = F.one_hot(target, num_classes=CLASS_NUM).float()
        weight = torch.clamp(weight, min=self.smoothing/(CLASS_NUM-1), max=1.-self.smoothing)
        loss = -weight * log_prob
        loss_tot = loss.sum(dim=-1).mean()
        return loss_tot

def train_proccess(network, part_data, part_label, proccess_no):
    
    loss_cal = LabelSmoothLoss()
    optimizer = optim.Adam(network.parameters())

    memory = ReplayMemory(network, loss_cal)
    memory.write(part_data, part_label)

    check_start = time.time()
    check_old = check_start
    save_counter = 0

    for epoch in range(EPOCHS):
        for step in range(int(len(memory.buffer)/BATCH_SIZE)):
            t1 = time.time()
            batch_image, batch_label = memory.sample(epoch)
            net_feature = network.forward(batch_image.clone().detach())
            if proccess_no == 0:
                arc_feature = network.arcface(net_feature, batch_label.clone().detach())
                loss_tot = loss_cal(arc_feature, torch.LongTensor(batch_label))
            elif proccess_no == 1:
                cos_feature = network.cosface(net_feature, batch_label.clone().detach())
                loss_tot = loss_cal(cos_feature, torch.LongTensor(batch_label))
            elif proccess_no == 2:
                sphere_feature = network.sphereface(net_feature, batch_label.clone().detach())
                loss_tot = loss_cal(sphere_feature, torch.LongTensor(batch_label))

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

            t2 = time.time()
            time_check = time.time()
            #for record
            interval = time_check - check_old 
            if interval > 20 and proccess_no==1:
                File = open("test.txt", "a")
                File.write(str(round(time_check - check_start,2))+"\t"+str(round(float(loss_tot.detach()),4))+"\n")
                check_old = time_check
                File.close()

            print('['+str(proccess_no)+'] ','steps:',step,'/',int(len(part_data)/BATCH_SIZE),'epoch:', epoch,\
                  '/',EPOCHS,'loss :',round(float(loss_tot.detach()),4), round(t2-t1,2),'[s/step]')
        
        torch.save(network, 'model_'+str(proccess_no)+'_'+str(save_counter)+'_'+str(epoch)+'.pt')



