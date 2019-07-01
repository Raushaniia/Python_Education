from __future__ import absolute_import
from __future__ import print_function

import random
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
import os
import numpy as np
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add
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
labels = np.zeros((len(train_images_names), IMG_HEIGHT, IMG_HEIGHT, 1), dtype=np.bool)

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


    mask_border = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    h_st = (IMG_HEIGHT - MASK_HEIGHT) // 2
    h_fin = MASK_HEIGHT + h_st
    w_st = (IMG_WIDTH - MASK_WIDTH) // 2
    w_fin = MASK_WIDTH + w_st
    if mask is not None:
        mask = cv2.resize(mask, (MASK_HEIGHT, MASK_WIDTH))
    mask_border[h_st:h_fin, w_st:w_fin] = mask
    labels[idx] = np.expand_dims(mask_border, axis=2)
    plt.imshow(mask_border)
    mask = np.expand_dims(cv2.resize(mask, (MASK_HEIGHT, MASK_WIDTH)), axis=2)
    #labels[idx] = mask.astype(np.bool)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def encoder_block(input_tensor, m, n):

    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    added_1 = _shortcut(input_tensor, x)

    x = BatchNormalization()(added_1)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    added_2 = _shortcut(added_1, x)

    return added_2

def decoder_block(input_tensor, m, n):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(1, 1))(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(1, 1))(x)

    return x

inputs = Input(shape=(576, 576, 1))
#import Image
#inputs = inputs.resize((256, 256), Image.NEAREST)      # use nearest neighbour

x = BatchNormalization()(inputs)
x = Activation('relu')(x)
x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

encoder_1 = encoder_block(input_tensor=x, m=64, n=64)

encoder_2 = encoder_block(input_tensor=encoder_1, m=64, n=128)

encoder_3 = encoder_block(input_tensor=encoder_2, m=128, n=256)

encoder_4 = encoder_block(input_tensor=encoder_3, m=256, n=512)

decoder_4 = decoder_block(input_tensor=encoder_4, m=512, n=256)

decoder_3_in = add([decoder_4, encoder_3])
decoder_3_in = Activation('relu')(decoder_3_in)

decoder_3 = decoder_block(input_tensor=decoder_3_in, m=256, n=128)

decoder_2_in = add([decoder_3, encoder_2])
decoder_2_in = Activation('relu')(decoder_2_in)

decoder_2 = decoder_block(input_tensor=decoder_2_in, m=128, n=64)

decoder_1_in = add([decoder_2, encoder_1])
decoder_1_in = Activation('relu')(decoder_1_in)

decoder_1 = decoder_block(input_tensor=decoder_1_in, m=64, n=64)

x = UpSampling2D((2, 2))(decoder_1)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=1, kernel_size=(2, 2), padding="same")(x)

model = Model(inputs=inputs, outputs=x)

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X_train, y_train,
                epochs=1,
                batch_size=1,
                shuffle=True,
                validation_data=(X_test, y_test))

def preprocess_batch(batch_X):
    batch_X_prep = (batch_X - 127.5) / 127.5
    return batch_X_prep

import pickle

saved_model = pickle.dump(model,open('model.txt', 'wb'))

# Load the pickled model
knn_from_pickle = pickle.load(open('model.txt','rb'))

# Use the loaded pickled model to make predictions
#knn_from_pickle.predict(X_test)

test_image = X_test[1]
ix = 0 #random
test_image_prep = np.expand_dims(test_image, axis=0)
test_mask = knn_from_pickle.predict(test_image_prep, batch_size=1)
test_mask = np.squeeze(test_mask)
test_mask = (test_mask*255).astype(np.uint8)
plt.imshow(np.squeeze(test_image), cmap='gray')
plt.imshow(test_mask, alpha=0.2);
plt.show()