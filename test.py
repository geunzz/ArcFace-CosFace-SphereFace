import torch
import numpy as np
from dataset_generation import data_generator
from model import ArcFace
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

datagen = data_generator(DATASET_PATH = 'C:/projects/dataset/thermal_image/80_60_test_dataset/', shuffle_sel=True)
data_class_set, data_array, label, data_name = datagen.data_label_set_gen()
arcface = torch.load('arcface_model_6.pt')

BATCH_SIZE = 5
THRESHOLD = 0.999
CLASS_NUM = 12

positive = 0
total = 0
plot_feature = []

for step in range(int(len(data_array)/BATCH_SIZE)):
    indices = range(step*BATCH_SIZE,step*BATCH_SIZE+BATCH_SIZE)
    batch_image = []
    batch_label = []
    for index in indices:
        sample = data_array[index]
        sample = np.transpose(sample)
        batch_image.append(sample)
        batch_label.append(label[index])
    prob_feature = arcface.test(torch.tensor(batch_image))
    prob_predict = torch.softmax(prob_feature, dim=1)
    predict = torch.argmax(prob_predict, dim=1).tolist()

    for i in range(BATCH_SIZE):
        plot_feature.append(prob_feature[i].tolist())
        #unseen class
        if batch_label[i] > CLASS_NUM:
            if prob_predict[i][predict[i]]<THRESHOLD:
                positive = positive + 1
                total = total + 1
            else:
                total = total + 1
        #known class
        elif (predict[i] == batch_label[i]) and (prob_predict[i][predict[i]]>THRESHOLD):
            positive = positive + 1
            total = total + 1
        else:
            total = total + 1

accuracy = 100*round(positive/total,2)
print('accuracy:',accuracy,'[%]')
