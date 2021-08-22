import torch
import numpy as np
from collections import Counter
from dataset_generation import data_generator

datagen = data_generator(DATASET_PATH = 'C:/projects/dataset/thermal_image/80_60_test_dataset/', shuffle_sel=True)
data_class_set, data_array, label, data_name = datagen.data_label_set_gen()
network = torch.load('model_2_0_19.pt')
network_sphere = torch.load('model_2_0_19.pt')

BATCH_SIZE = 5
THRESHOLD = 0.9
CLASS_NUM = 12
#for temperature scaling
ARC_SCALING = 5
COS_SCALING = 10
SPHERE_SCALING = 5

arc_positive = 0
cos_positive = 0
sphere_positive = 0
tot_positive = 0
total = 0
arc_total = 0
cos_total = 0
sphere_total = 0
plot_arc = []
plot_cos = []

def each_accuracy(batch_label, prob, predict, positive, total):
    pred_label = []
    pred_conf = []
    for i in range(BATCH_SIZE):
        #unseen class
        if batch_label[i] > CLASS_NUM:
            if prob[i][predict[i]] < THRESHOLD:
                positive = positive + 1
                total = total + 1
            else:
                total = total + 1
        #known class
        elif (predict[i] == batch_label[i]) and (prob[i][predict[i]]>THRESHOLD):
            positive = positive + 1
            total = total + 1
        else:
            total = total + 1
        
        pred_label.append(predict[i])
        pred_conf.append(prob[i][predict[i]])

    return positive, total, pred_label, pred_conf

def tot_accuracy(batch_label, tot_positive, total, \
                 arc_pred_label, arc_pred_conf, cos_pred_label, cos_pred_conf, sphere_pred_label, sphere_pred_conf):

    for i in range(len(batch_label)):
        total_prob = (arc_pred_conf[i]+cos_pred_conf[i]+sphere_pred_conf[i])/3
        conf_list = torch.tensor([arc_pred_conf[i], cos_pred_conf[i], sphere_pred_conf[i]])
        # total_prob = torch.median(conf_list)
        pred_list = [arc_pred_label[i], cos_pred_label[i], sphere_pred_label[i]]
        cnt = Counter(pred_list)
        final_pred = cnt.most_common(1)[-1]
        #unseen class
        if batch_label[i] > CLASS_NUM:
            #confidence가 낮으면 unseen class를 분류해 낸 것으로 간주
            if total_prob < THRESHOLD:
                tot_positive = tot_positive + 1
                total = total + 1
            #voting으로 다수결 판단 (과반수 이상이면)
            elif final_pred[1] < len(final_pred)/2:
                tot_positive = tot_positive + 1
                total = total + 1
            else:
                total = total + 1
        #knwon class
        else:
            if (final_pred[0] == batch_label[i]) and (final_pred[1] > len(final_pred)/2) and (total_prob>THRESHOLD):
                tot_positive = tot_positive + 1
                total = total + 1
            else:
                total = total + 1

    return tot_positive, total

total_label = []
for step in range(int(len(data_array)/BATCH_SIZE)):
    indices = range(step*BATCH_SIZE,step*BATCH_SIZE+BATCH_SIZE)
    batch_image = []
    batch_label = []
    for index in indices:
        sample = data_array[index]
        sample = np.transpose(sample)
        batch_image.append(sample)
        batch_label.append(label[index])
        total_label.append(label[index])
    arc_feature, cos_feature, sphere_feature = network.test(torch.tensor(batch_image))
    # _, _, sphere_feature = network_sphere.test(torch.tensor(batch_image))

    arc_prob = torch.softmax(arc_feature/ARC_SCALING, dim=1)
    cos_prob = torch.softmax(cos_feature/COS_SCALING, dim=1)
    sphere_prob = torch.softmax(sphere_feature/SPHERE_SCALING, dim=1)
    arc_predict = torch.argmax(arc_prob, dim=1).tolist()
    cos_predict = torch.argmax(cos_prob, dim=1).tolist()
    sphere_predict = torch.argmax(sphere_prob, dim=1).tolist()

    arc_positive, arc_total, arc_pred_label, arc_pred_conf = \
        each_accuracy(batch_label, arc_prob, arc_predict, arc_positive, arc_total)
    cos_positive, cos_total, cos_pred_label, cos_pred_conf = \
        each_accuracy(batch_label, cos_prob, cos_predict, cos_positive, cos_total)
    sphere_positive, sphere_total, sphere_pred_label, sphere_pred_conf = \
        each_accuracy(batch_label, sphere_prob, sphere_predict, sphere_positive, sphere_total)    
    tot_positive, total = \
        tot_accuracy(batch_label, tot_positive, total, \
                     arc_pred_label, arc_pred_conf, cos_pred_label, cos_pred_conf, sphere_pred_label, sphere_pred_conf)

arc_accuracy = 100*round(arc_positive/arc_total,2)
cos_accuracy = 100*round(cos_positive/cos_total,2)
sphere_accuracy = 100*round(sphere_positive/sphere_total,2)
total_accuracy = 100*round(tot_positive/total,2)

print('arc_accuracy:',arc_accuracy,'[%]')
print('cos_accuracy:',cos_accuracy,'[%]')
print('sphere_accuracy:',sphere_accuracy,'[%]')
print('total_accuracy:',total_accuracy,'[%]')


