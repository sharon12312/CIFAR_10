from keras.models import load_model
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from skimage.io import imsave
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from PIL import Image
import os
import h5py
import sys

PATH = './Data/Images/'
MODEL_PATH = './Models/model.tf.hdf5'

categories = ["phlox", "rose", "calendula", "iris", "leucanthemum maximum",
              "bellflower", "viola", "rudbeckia laciniata (Goldquelle)", "peony", "aquilegia"]

model = load_model(MODEL_PATH)
print('Model was loaded..')
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4),
              metrics=['accuracy'])

for filename in os.listdir(PATH):
    img = Image.open(PATH + filename)
    new_img = img.resize((128, 128), Image.ANTIALIAS)
    new_img = np.reshape(new_img,[1,128,128,4])
    image = np.array(new_img, dtype=float)

    classes = model.predict_classes(new_img)
    x = classes[0]
    print(categories[x])