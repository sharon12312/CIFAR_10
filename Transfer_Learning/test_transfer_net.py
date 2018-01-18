from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pandas as pd
import numpy as np

# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = './Data/Train'
validation_data_dir = './Data/Test'
nb_train_samples = 210
nb_validation_samples = 19

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
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])

# Train New Model
train = pd.read_csv("./Data/Train/flower_labels.csv")
test = pd.read_csv("./Data/Test/flower_labels.csv")
train_path = "./Data/Train/"
test_path = "./Data/Test/"

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

# load the weights that yielded the best validation accuracy
model_final.load_weights('./Models/model.tf.hdf5')

# evaluate test accuracy
score = model_final.evaluate(x_test, test_y, verbose=0)
accuracy = 100 * score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)