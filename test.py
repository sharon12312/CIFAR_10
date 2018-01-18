from __future__ import print_function
import tensorflow as tf
from PIL import Image
from keras.models import load_model
import keras
import os
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import multi_gpu_model
import argparse
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

MODEL_PATH = './Models/'
model_name = 'keras_cifar10_trained_model_cnn_1.h5'
model_path = os.path.join(MODEL_PATH, model_name)

# Load 5 classes: dog, cat, frog, horse and bird.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
shrink_data = True

if shrink_data:
    selected_classes = [2, 3, 5, 6, 7]
    print('train\n', x_train.shape, y_train.shape)
    x = [ex for ex, ey in zip(x_train, y_train) if ey in selected_classes]
    y = [ey for ex, ey in zip(x_train, y_train) if ey in selected_classes]
    x_train = np.stack(x)
    y_train = np.stack(y).reshape(-1,1)
    print(x_train.shape, y_train.shape)

    print('test\n', x_test.shape, y_test.shape)
    x = [ex for ex, ey in zip(x_test, y_test) if ey in selected_classes]
    y = [ey for ex, ey in zip(x_test, y_test) if ey in selected_classes]
    x_test = np.stack(x)
    y_test = np.stack(y).reshape(-1,1)
    print(x_test.shape, y_test.shape)
else:
    print('train\n', x_train.shape, y_train.shape)
    print('test\n', x_test.shape, y_test.shape)

Load_model = True
exist = False
if Load_model and os.path.exists(model_path):
    print('==> loading pre-trained model')
    model = load_model(model_path)
    exist = True
else:
    print('Model doesn\'t exists.')

# one-hot encode the labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Score trained model.
if(exist):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('==> Test loss:', scores[0])
    print('==> Test accuracy:', scores[1])