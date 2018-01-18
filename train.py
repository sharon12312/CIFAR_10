from __future__ import print_function
import keras
from keras.models import Model
from keras import backend as K
from os.path import join as pathjoin
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.models import load_model
from keras.utils import multi_gpu_model
import argparse
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from keras.applications.inception_v3 import InceptionV3

MODEL_PATH = './Models/'

# Visualization
def weights_to_visualize(weights_layer):
    x1w = weights_layer.get_weights()[0][:, :, 0, :]
    for i in range(1, 26):
        plt.subplot(5, 5, i)
        plt.imshow(x1w[:, :, i], interpolation="nearest", cmap="gray")
    plt.show()

# Load 5 classes: dog, cat, frog, horse and bird.
shrink_data = True

# Load Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Get 5 classes
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

# Parameters for training Model
batch_size = 50
num_classes = 10
epochs = 10
eta = 0.0001

# Paths for saving models
Load_model = True
save_dir = MODEL_PATH
model_name = 'keras_cifar10_trained_model_cnn_1.h5'
model_path = os.path.join(save_dir, model_name)
model_checkpoint_call_back = keras.callbacks\
                .ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build neural network for training
if Load_model and os.path.exists(model_path):
    print('==> loading pre-trained model')
    model = load_model(model_path)
    model.net_name = 'mymodel'

else:
    print('==> creating a new model')
    model = Sequential()
    model.net_name = 'mymodel'

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

# initiate optimizer
# opt = keras.optimizers.rmsprop(lr=eta, decay=1e-6)
opt = keras.optimizers.adam(lr=eta, decay=1e-6)


# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Shape image train by dividing the train & test by 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

### Train Model
for epoch in range(epochs):
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1,
              verbose=1, shuffle=True, validation_data=(x_test, y_test))

    fname = pathjoin(MODEL_PATH, 'new-epoch%d-%s' % (epoch + 33, model.net_name))
    print('saving model to: %s' % fname)
    model.save_weights(fname, overwrite=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model.save(model_path)
print('==> Saved trained model at %s ' % model_path)

# Print Params & write to Log file
print('---------------------')
print('Model - Parameters:')
print('Batch Size: ', batch_size)
print('Number of epoches: ', epochs)
print('Learning reat: ', eta)
print('---------------------')

# Write parameters to a local file
with open('./Logs/params.txt', 'a') as the_file:
    the_file.write('Train:'  + '\n')
    the_file.write('Batch Size: ' +  str(batch_size) + '\n')
    the_file.write('Number of epoches: ' + str(epochs) + '\n')
    the_file.write('Learning reat: ' + str(eta) + '\n')
    the_file.write('----------------\n')