import flask
from flask import send_file
from flask import request, json
import os
from flask import Flask, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pickle
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
import matplotlib
import PIL
from PIL import Image


UPLOAD_FOLDER = 'UPLOAD_FOLDER/'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/test', methods=['GET', 'PUT', 'OPTIONS','POST'])
def home():
    if request.method == 'POST':
        file = flask.request.files['photo']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            segment_file()
    return "<h1>Image uploaded.</h1>"

#return render_template("hello.html")

@app.route('/get_image', methods=['GET', 'PUT', 'OPTIONS'])
def get_image():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route("/getjson", methods=['POST'])
def getjson():
    try:
        request_json = flask.request.get_json()
        response = flask.jsonify(request_json)
        response.status_code = 200
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content": exception_message})
        response.status_code = 400
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


IMG_WIDTH = 576
IMG_HEIGHT = 576
MASK_WIDTH = 388
MASK_HEIGHT = 388
TRAIN_PATH = "data/"
IMG_PATH = 'images1/'
MASK_PATH = 'masks1/'

def segment_file():
    train_images_names = [file for file in os.listdir(TRAIN_PATH + MASK_PATH) if '.png' in file or '.jpg' in file]
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
        # labels[idx] = mask.astype(np.bool)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    # Load the pickled model
    knn_from_pickle = pickle.load(open('model.txt','rb'))
    test_image = X_test[1]
    ix = 0  # random
    test_image_prep = np.expand_dims(test_image, axis=0)
    test_mask = knn_from_pickle.predict(test_image_prep, batch_size=1)
    test_mask = np.squeeze(test_mask)
    test_mask = (test_mask * 255).astype(np.uint8)
    plt.imshow(np.squeeze(test_image), cmap='gray')
    plt.imshow(test_mask, alpha=0.2);
    plt.show()

    matplotlib.image.imsave(os.path.join(UPLOAD_FOLDER, 'mask.png'), test_mask)
    matplotlib.image.imsave(os.path.join(UPLOAD_FOLDER, 'image.png'), np.squeeze(test_image))

    background = Image.open(os.path.join(UPLOAD_FOLDER, 'mask.png'))
    overlay = Image.open(os.path.join(UPLOAD_FOLDER, 'image.png'))

    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(os.path.join(UPLOAD_FOLDER, 'new.png'), "PNG")

@app.route('/uploadimage', methods=['GET', 'POST', 'PUT'])
def upload_file():
    if request.method == 'PUT':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('/get_image',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

app.run()