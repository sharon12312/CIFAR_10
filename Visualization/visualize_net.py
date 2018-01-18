import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
import theano
from keras import backend as K
import numpy as np
from keras.datasets import cifar10
from scipy.misc import imread
from PIL import Image, ImageFilter

fname = '../Models/new-epoch%d-mymodel'

# Load datatset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def get_model(weights_fname, epoch):
    # build the net
    model = Sequential()
    model.net_name = 'mymodel'

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    if epoch > 0:
        model.load_weights(weights_fname % epoch)

    return model

def get_fig_size(num_filters):
    # get the image dimensions for different number of conv filters
    if num_filters == 32:
        return 4, 8,  # 4X8 images == 32 filters

    if num_filters == 64:
        return 8, 8,  # 8X8 images == 64 filters

    if num_filters == 96:
        return 12, 8,  # 12X8 images == 96 filters

    if num_filters == 30:
        return 4, 8,

    if num_filters == 15:
        return 4, 4,

    raise ValueError('bad num_filters=%s' % num_filters)

def get_fig_dim(num_filters):
    if num_filters == 32:
        return 16, 8,  # 4X8 images == 32 filters

    if num_filters == 64:
        return 16, 16,  # 8X8 images == 64 filters

    if num_filters == 96:
        return 16, 24,  # 8X8 images == 96 filters

    if num_filters == 30:
        return 4, 8,

    if num_filters == 15:
        return 4, 4,

    raise ValueError('bad num_filters=%s for dim' % num_filters)

def plot_layer(convolutions):
    # show the 32 convolutions of the image @ conv
    dim = get_fig_dim(convolutions[0].shape[1])
    fig = plt.figure(figsize=dim)

    # disable spacing between images
    gs1 = gridspec.GridSpec(*dim)  # 16,8 for 32
    gs1.update(wspace=0.000025, hspace=0.00005)

    for i, convolution in enumerate(convolutions[0]):
        fig_size = get_fig_size(convolutions[0].shape[1])
        fig_size = fig_size + (i + 1, )
        a = fig.add_subplot(*fig_size)
        imgplot = plt.imshow(convolution[0], cmap=cm.Greys_r)
        imgplot.axes.axis('off')
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')

    fig.tight_layout()
    return fig

def conv_img_with_layer(model, img, layer_num):
    # get all activation layers
    convout_layers = [layer for layer in model.layers
                      if isinstance(layer, Activation)]

    print('got %d activation layers' % len(convout_layers))

    # get only the output up to the requested layer
    layer_output = convout_layers[layer_num].output
    convout1_f = K.function([model.input], [layer_output])

    # get the image @ convolved with the 32 filters
    convolutions = convout1_f([img.reshape(-1, 32, 32, 3)])

    return convolutions

def visualize(epoch, layer):
    # load the model and weights
    model = get_model(fname, epoch)

    # Load 1 image from CIFAR10 train to visualize
    img = x_train[0]

    # hook the conv layer
    convolutions = conv_img_with_layer(model, img, layer)

    fig = plot_layer(convolutions)
    fig.savefig('./Layer-%d_epoch-%d.png' % (layer, epoch), bbox_inches='tight')

    del model

if __name__ == '__main__':
    for l in range(28):
        for e in range(2):
            print('layer %d epoch %d' % (l, e))
        visualize(l, e)
