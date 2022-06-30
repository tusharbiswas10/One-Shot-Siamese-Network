import pickle
import cv2
import os.path
import os
#from file_with_methods import create_model
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
#from tensorflow.python.keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
#from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from tensorflow.python.keras.layers import Layer
from keras import backend as K
from keras.models import load_model
K.set_image_data_format('channels_first')
from keras_facenet import FaceNet

def converter(ip, model):
    pic = cv2.imread(ip, 1)
    pic = cv2.resize(pic, (160,160))
    xt = np.array([pic])
    code = model.predict_on_batch(xt)
    return code


data = (3, 160, 160)
paths="/content/drive/MyDrive/img"
#
faces = []

images = {}


def refactor(batch_size=16):
    y = np.zeros((batch_size, 2, 1))
    
    positives = np.zeros((batch_size, data[0], data[1], data[2]))
    base = np.zeros((batch_size, data[0], data[1], data[2]))
    negatives = np.zeros((batch_size, data[0], data[1], data[2]))

    while True:
        for i in range(batch_size):
            positiveFace = faces[np.random.randint(len(faces))]
            negativeFace = faces[np.random.randint(len(faces))]
            while positiveFace == negativeFace:
                negativeFace = faces[np.random.randint(len(faces))]

            positives[i] = images[positiveFace][np.random.randint(len(images[positiveFace]))]
            base[i] = images[positiveFace][np.random.randint(len(images[positiveFace]))]
            negatives[i] = images[negativeFace][np.random.randint(len(images[negativeFace]))]

        x_data = {'anchor': base,
                  'anchorPositive': positives,
                  'anchorNegative': negatives
                  }

        yield (x_data, [y, y, y])

refactor()

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[1])) )
    neg_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[2])) )
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.maximum(basic_loss, 0.0)
   
    return loss



fModel = load_model('vnv/model/facenet_keras.h5', custom_objects={'triplet_loss': triplet_loss})
fModel.compile(loss=triplet_loss)




def dbManager():
    # check for existing database
    if os.path.exists('database/uDb.pickle'):
        with open('database/uDb.pickle', 'rb') as handle:
            db = pickle.load(handle)   
    else:
        # make a new one
        # we use a dict for keeping track of mapping of each person with his/her face encoding
        db = {}
        # create the directory for saving the db pickle file
        os.makedirs('database',exist_ok=True)
        with open('database/uDb.pickle', 'wb') as handle:
            pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    return db

# adds a new user face to the database using his/her image stored on disk using the image path
def addImg(db, fModel, name, img_path):
    if name not in db: 
        db[name] = converter(img_path, fModel)
        print("Encodings:",db[name])
        # save the database
        with open('database/uDb.pickle', 'wb') as handle:
                pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('User ' + name + ' is added')
    else:
        print('This name is already in database.')

db = dbManager()

addImg(db, fModel, "john", "vnv\sample\john.jpg")

# recognize the input user face encoding by checking for it in the database
def look_up(image_path, database, model, threshold = 0.6):
    # find the face encodings for the input image
    print(image_path)
    encoding = converter(image_path,model)
    
    min_dist = 99999
    # loop over all the recorded encodings in database 
    for name in database:
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[name], encoding) )
        # check if minimum distance or not
        if dist < min_dist:
            min_dist = dist
            identity = name
    print( "Identity:",identity)
    
        
    return min_dist, identity


# takes an input image and performs face recognition on it
def getFace(db, fModel, threshold = 0.7, img_loc = "vnv/templates/saved_image/temp.jpg"):
    # resize the image
    img = cv2.imread(img_loc, 1)
    img = cv2.resize(img, (96, 96))
    # save the temporary image
    cv2.imwrite("vnv/templates/saved_image/temp.jpg", img)

    look_up("vnv/templates/saved_image/temp.jpg", db, fModel, threshold)


# test 1
getFace(db,fModel,  threshold = 0.7, img_loc = "/content/drive/MyDrive/sample pic/john-cena.jpg")