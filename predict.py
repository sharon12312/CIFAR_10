from keras.models import load_model
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.io import imsave
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from PIL import Image
import os
import h5py
import sys

# PATH = './Images'
PATH = sys.argv[1]
MODEL_PATH = './Models/keras_cifar10_trained_model_cnn_1.h5'

categories = ["X", "X", "bird", "cat", "X", "dog", "frog", "horse", "X", "X"]

model = load_model(MODEL_PATH)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

for filename in os.listdir(PATH):
    img = Image.open(sys.argv[1] + filename)
    new_img = img.resize((32, 32), Image.ANTIALIAS)
    new_img = np.reshape(new_img,[1,32,32,3])
    image = np.array(new_img, dtype=float)

    classes = model.predict_classes(new_img)
    x = classes[0]
    print(categories[x])