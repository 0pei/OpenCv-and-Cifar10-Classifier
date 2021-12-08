import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from GUI import Ui_MainWindow
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.batchSize=100
        self.learning=0.001
        self.optimizer="Adadelta"
        self.ui.teChooseData.setText("5")
        self.ui.btnShowTrainImages.clicked.connect(self.btnShowTrainImagesClicked)
        self.ui.btnShowHyperParameter.clicked.connect(self.btnShowHyperParameterClicked)
        self.ui.btnShowModelShortcut.clicked.connect(self.btnShowModelShortcutClicked)
        self.ui.btnShowAccuracy.clicked.connect(self.btnShowAccuracyClicked)
        self.ui.btnTest.clicked.connect(self.btnTestClicked)
        self.class_name = {
            0: 'plane', # airplane
            1: 'car',   # automobile
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck',
        }
        self.model = Sequential([
            Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(1000, activation='softmax')
        ])
        
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict=pickle.load(fo, encoding='bytes')
        return dict
    # 5.1
    def btnShowTrainImagesClicked(self):
        data_batch=[]
        data_batch.append(self.unpickle('.\cifar-10-batches-py\data_batch_1'))
        data_batch.append(self.unpickle('.\cifar-10-batches-py\data_batch_2'))
        data_batch.append(self.unpickle('.\cifar-10-batches-py\data_batch_3'))
        data_batch.append(self.unpickle('.\cifar-10-batches-py\data_batch_4'))
        data_batch.append(self.unpickle('.\cifar-10-batches-py\data_batch_5'))
        batches=self.unpickle('.\cifar-10-batches-py\\batches.meta')
        img=[]
        for i in range(9):
            x=random.randint(0,4)
            data=random.randint(0,9999)
            img.append(data_batch[x][b'data'][data].reshape(3,32,32).transpose(1,2,0))
            plt.subplot(3,3,i+1)
            plt.imshow(img[i])
            plt.title(batches[b'label_names'][data_batch[x][b'labels'][data]].decode('utf-8'))
            plt.axis('off')
        plt.show()
    # 5.2
    def btnShowHyperParameterClicked(self):
        print("hyperparameter:")
        print("batch size:", self.batchSize)
        print("learning rate:", self.learning)
        print("optimizer:", self.optimizer)
    # 5.3
    def btnShowModelShortcutClicked(self):
        print(self.model.summary())
    # 5.4
    def btnShowAccuracyClicked(self):
        cv2.imshow('Accuracy', cv2.imread('.\\accuracy.jpg'))
        cv2.imshow('Loss', cv2.imread('.\loss.jpg'))
    # 5.5
    def btnTestClicked(self):
        model=load_model('.\VGG16.h5')
        testImageNo=int(self.ui.teChooseData.toPlainText())
        test_batch=self.unpickle('.\cifar-10-batches-py\\test_batch')
        img=test_batch[b'data'][testImageNo].reshape(3,32,32).transpose(1,2,0)
        plt.figure(2)
        plt.imshow(img)
        img=img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        feature=model.predict(img)
        # index = np.argmax(feature, axis=1)
        # print(self.class_name[index[0]])
        x = np.arange(len(self.class_name.items()))
        plt.figure(3)
        plt.bar(x, feature[0])
        plt.xticks(x, self.class_name.values())
        plt.show()
        
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())