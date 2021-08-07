import os
import cv2
import numpy as np
import random

# using way
# datagen = data_generator(DATASET_PATH = 'PATH/TO/THE/IMAGE DATA', shuffle_sel=True)
# data_class_set, data_array, label, data_name = datagen.data_label_set_gen()
# x_train, x_test, y_train, y_test, z_train, z_test = datagen.train_val_split(data_array, label, data_name, test_prob=0.2)

class data_generator():

    def __init__(self, DATASET_PATH, shuffle_sel):
        self.dataset_path = DATASET_PATH
        self.data_class_set=[]
        self.class_id_matching = []
        self.id_val = 0
        self.X = []
        self.Y = []
        self.Z = []
        self.shuffle_sel = shuffle_sel

    def data_label_set_gen(self, shuffle_sel=True):
        for class_name in os.listdir(self.dataset_path):
            self.class_id_matching = [class_name, self.id_val]
            class_path = self.dataset_path + class_name
            for data in os.listdir(class_path):
                data_path = class_path +  '/' + data
                image = cv2.imread(data_path)
                data_class = [image, data, self.id_val]
                self.data_class_set.append(data_class)

            self.id_val = self.id_val + 1
        if self.shuffle_sel == shuffle_sel:    
            random.shuffle(self.data_class_set)

        for i in range(0, len(self.data_class_set)):
            self.X.append(self.data_class_set[i][0])
            self.Y.append(self.data_class_set[i][2])
            self.Z.append(self.data_class_set[i][1])
        #self.data_class_set : [data array, name of data, label]
        return self.data_class_set, self.X, self.Y, self.Z 

    def train_val_split(self, data_array, label, data_name=[], test_prob=0.2):

        if len(data_array) == len(label):
            x_train = data_array[0:int(len(data_array)*(1 - test_prob))]
            x_test = data_array[int(len(data_array)*(1 - test_prob)):]
            
            y_train = label[0:int(len(label)*(1 - test_prob))]
            y_test = label[int(len(data_array)*(1 - test_prob)):]
            if data_name != []:
                z_train = data_name[0:int(len(data_name)*(1 - test_prob))]
                z_test = data_name[int(len(data_array)*(1 - test_prob)):]
            else:
                z_train = []
                z_test = []
        else:
            raise('Check the length match of dataset matrix with class index matrix.')

        x_train = np.array(x_train) #data array for training
        x_test = np.array(x_test) #data array for test
        y_train = np.array(y_train) #label array for training
        y_test = np.array(y_test) #label array for test
        z_train = np.array(z_train) #data name for training
        z_test = np.array(z_test) #data name for test
        
        return x_train, x_test, y_train, y_test, z_train, z_test











