# ArcFace
![arcface](https://user-images.githubusercontent.com/84235639/128590143-a6b9de0a-e123-409b-855c-a9ce73f28ffa.JPG)    
    
Arcface is a method of embedding feature vectors of images based on angles. Therefore, embeddings are distributed at specific angles for each class, and it is said that there is an advantage of embedding more precisely than the existing various metric learning methods. Before training, hyperparameters defined in uppercase letters in the train.py file must be specified in advance. The number of classes, batch size, and number of epochs must be entered, and the path of the data to be learned must also be entered.

    CLASS_NUM = 12
    EPOCHS = 30
    BATCH_SIZE = 16
    datagen = data_generator(DATASET_PATH = 'PATH/TO/THE/TRAIN_IMAGE/', shuffle_sel=True)

train.py trains the network containing the arcface algorithm, and creates a pytorch(.pt) file as the result.

    python train.py    
    
Similarly, in the test.py file, several hyperparameters must be specified in advance. The number of classes, batch size, and threshold value should be specified during training. The threshold value becomes the decision threshold to be considered correct when the model's confidence is greater than or equal to this value. If you do not want to use this value, simply specify 0 to disable the function.

    BATCH_SIZE = 5
    THRESHOLD = 0.9
    CLASS_NUM = 12
    datagen = data_generator(DATASET_PATH = 'PATH/TO/THE/TEST_IMAGE/', shuffle_sel=True)

After specifying the value, you can evaluate the performance of the trained model by executing the test.py file.

    python test.py

