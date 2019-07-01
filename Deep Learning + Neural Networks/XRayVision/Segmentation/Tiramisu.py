from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Input, concatenate, Activation, Dropout, \
    BatchNormalization

import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook as tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import os
import numpy as np

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add
from keras.layers.core import Flatten, Reshape
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Set some parameters
IMG_WIDTH = 576
IMG_HEIGHT = 576
MASK_WIDTH = 388
MASK_HEIGHT = 388
TRAIN_PATH = "data/"
IMG_PATH = 'images1/'
MASK_PATH = 'masks1/'

train_images_names = [file for file in os.listdir(TRAIN_PATH+MASK_PATH) if '.png' in file or '.jpg' in file]
images = np.zeros((len(train_images_names), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
labels = np.zeros((len(train_images_names), MASK_HEIGHT, MASK_WIDTH, 1), dtype=np.bool)

for idx, filename in tqdm(enumerate(train_images_names)):
    img = cv2.imread(TRAIN_PATH + IMG_PATH + filename, 0)
    if img is not None:
        img = cv2.resize(img, (MASK_HEIGHT, MASK_WIDTH))
    img_border = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    h_st = (IMG_HEIGHT - MASK_HEIGHT) // 2
    h_fin = MASK_HEIGHT + h_st
    w_st = (IMG_WIDTH - MASK_WIDTH) // 2
    w_fin = MASK_WIDTH + w_st
    img_border[h_st:h_fin, w_st:w_fin] = img
    images[idx] = np.expand_dims(img_border, axis=2)
    plt.imshow(img_border)
    mask = cv2.imread(TRAIN_PATH + MASK_PATH + filename, 0)
    mask = np.expand_dims(cv2.resize(mask, (MASK_HEIGHT, MASK_WIDTH)), axis=2)
    labels[idx] = mask.astype(np.bool)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def layer(in_layer, filters):
    bn = BatchNormalization()(in_layer)
    relu = Activation('relu')(bn)
    conv = Conv2D(filters, kernel_size=(3, 3), padding='same')(relu)
    drop = Dropout(0.2)(conv)
    return drop


def transition_down(in_layer):
    filters = int(in_layer.shape[-1])
    bn = BatchNormalization()(in_layer)
    relu = Activation('relu')(bn)
    conv = Conv2D(filters, kernel_size=(1, 1), padding='same')(relu)
    drop = Dropout(0.2)(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(drop)
    return pool


def transition_up(in_layer):
    filters = int(in_layer.shape[-1])
    updoot = Conv2DTranspose(filters, (3, 3), strides=2, padding='same')(in_layer)
    return updoot


def dense_block(in_layer, filters, num_layers=4):
    layer_output = [in_layer, layer(in_layer, filters)]
    for _ in range(1, num_layers):
        concat = concatenate(layer_output[-2:], axis=-1)
        out = layer(concat, filters)
        layer_output.append(out)
    concat = concatenate(layer_output[1:], axis=-1)
    return concat


def get_layers(in_layer, growth_size, depth, dense_layers):
    if depth == 1:
        return dense_block(in_layer, growth_size, dense_layers[0])
    densedown = dense_block(in_layer, growth_size, dense_layers[0])
    catdown = concatenate([in_layer, densedown], axis=-1)
    td = transition_down(catdown)
    deep = get_layers(td, growth_size, depth - 1, dense_layers[1:])
    tu = transition_up(deep)
    catup = concatenate([tu, catdown], axis=-1)
    denseup = dense_block(catup, growth_size, dense_layers[0])
    return denseup


def Tiramisu(in_shape=(576, 576, 1), num_classes=1, depth=6, growth_size=16, dense_layers=[4, 5, 7, 10, 12, 15]):
    filters = growth_size * 3
    if type(dense_layers) != list:
        dense_layers = [dense_layers for _ in range(depth)]
    inputs = Input(in_shape)
    conv_in = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)

    tiramisu = get_layers(conv_in, growth_size, depth, dense_layers)

    conv_out = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(tiramisu)
    return Model(inputs=[inputs], outputs=[conv_out])

model = Tiramisu(in_shape=(576, 576, 1), num_classes=1, depth=6, growth_size=16, dense_layers=[4, 5, 7, 10, 12, 15])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X_train, X_train,
                epochs=1,
                batch_size=1,
                shuffle=True,
                validation_data=(X_test, X_test))

def preprocess_batch(batch_X):
    batch_X_prep = (batch_X - 127.5) / 127.5
    return batch_X_prep


test_image = X_test[1]
ix = 0 #random
test_image_prep = preprocess_batch(test_image)
test_image_prep = np.expand_dims(test_image_prep, axis=0)
test_mask = model.predict(test_image_prep, batch_size=1)
test_mask = np.squeeze(test_mask)
test_mask = (test_mask*255).astype(np.uint8)
f = plt.figure(figsize=(10,10))
h_st = (IMG_HEIGHT - MASK_HEIGHT) // 2
h_fin = MASK_HEIGHT + h_st
w_st = (IMG_WIDTH - MASK_WIDTH) // 2
w_fin = MASK_WIDTH + w_st
plt.imshow(np.squeeze(test_image), cmap='gray')
plt.imshow(test_mask, alpha=0.2);
plt.show()

