from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import pandas as pd
import os
import numpy as np

# dimensions of our images.
model_path = './Transfer_Learning/Models/model.tf.hdf5'
img_width, img_height = 128, 128
nb_train_samples = 210
nb_validation_samples = 19
epochs = 20
batch_size = 5
Load_model = True

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
print('Model loaded.')

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=1e-4), metrics=["accuracy"])

# Build neural network for training
if Load_model and os.path.exists(model_path):
    print('==> loading pre-trained model')
    model_final = load_model(model_path)
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=1e-4), metrics=["accuracy"])

# Train New Model
train = pd.read_csv("./Transfer_Learning/Data/Train/flower_labels.csv")
test = pd.read_csv("./Transfer_Learning/Data/Test/flower_labels.csv")
train_path = "./Transfer_Learning/Data/Train/"
test_path = "./Transfer_Learning/Data/Test/"

# Convert Images from path to x_train, y_train, x_test, y_test
x_train = []
for i in range(len(train)):
    temp_img=image.load_img(train_path + train['file'][i],target_size=(img_width,img_height))
    temp_img=image.img_to_array(temp_img)
    x_train.append(temp_img)

#converting train images to array and applying mean subtraction processing
x_train=np.array(x_train)
x_train=preprocess_input(x_train)

x_test = []
for i in range(len(test)):
    temp_img=image.load_img(test_path + test['file'][i],target_size=(img_width,img_height))
    temp_img=image.img_to_array(temp_img)
    x_test.append(temp_img)

#converting train images to array and applying mean subtraction processing
x_test = np.array(x_test)
x_test = preprocess_input(x_test)

# one-hot encoding for the target variable - for train
train_y = np.asarray(train['label'])
train_y = pd.get_dummies(train_y)
train_y = np.array(train_y)

# one-hot encoding for the target variable - for test
test_y = np.asarray(test['label'])
test_y = pd.get_dummies(test_y)
test_y = np.array(test_y)

checkpointer = ModelCheckpoint(filepath='./Transfer_Learning/Models/model.tf.hdf5', verbose=1, save_best_only=True)
model_final.fit(x_train, train_y, batch_size=batch_size, epochs=epochs,
          validation_split=0.2, callbacks=[checkpointer],
          verbose=1, shuffle=True)