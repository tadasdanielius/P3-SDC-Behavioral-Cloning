import pandas as pd
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
import math

import keras
from keras.optimizers import Adam, SGD
import json

import app.config as cfg
import app.transform as t
import app.iterator as it
import app.model as m

# Take left image and slightly push camera to the right
left = lambda row: (row['left'][0].strip(), row['steering'][0] + cfg.SHIFT_CAMERA)

# Take right image and slightly push camera to the left
center = lambda row: (row['center'][0].strip(), row['steering'][0])

# Take center image. No need to adjust camera
right = lambda row: (row['right'][0].strip(), row['steering'][0] - cfg.SHIFT_CAMERA)

# List of all cameras. Later will randomly pick camera
cameras = [left, right, center]

# Load training log
def load_data(fn='driving_log.csv'):
    return pd.read_csv(fn)

def take_random_image(row):
    # Pick random camera
    camera = cameras[np.random.randint(len(cameras))]
    fn, angle = camera(row)

    # Load image 
    if cfg.VERBOSE:
        print('loading image {}'.format(fn))

    img= load_image(fn)

    # Augment the image
    img, angle = t.augment_image(img, angle)

    return img, angle


# Take image from training log by index
take = lambda log, idx: take_random_image( log.iloc[[idx]].reset_index() )

@it.threadsafe_generator
def batch_generator(driving_log, batch_size = cfg.BATCH_SIZE):
    
    x_images = np.zeros((batch_size, cfg.IMG_RESIZE_SHAPE[0], cfg.IMG_RESIZE_SHAPE[1], 3))
    labels = np.zeros(batch_size)
    
    while 1:
        for idx in range(batch_size):
            record = driving_log.iloc[[np.random.randint(len(driving_log))]].reset_index()
            
            # Here is where we are going to regenerate images which has 
            # low value of steering angle, since we need to balance out
            # training images with different steering angle values
            invalid = True
            while invalid:
                x, y = take_random_image(record)
                if abs(y) < cfg.PROB_RANGE:
                    invalid = False if np.random.uniform() > cfg.THRESHOLD else True
                else:
                    invalid = False
            
            x_images[idx] = x
            labels[idx] = y
        yield x_images, labels

def sample_count(array_size, batch_size):
    return math.ceil(array_size / batch_size) * batch_size

def fit_model(log, nb_epoch=3, model=None):
    train, validation = train_test_split(log, test_size=0.10)
    print('Training: {}\nValidation: {}'.format(train.shape, validation.shape))
    
    model = model if model != None else m.get_model()
    model.summary()
    model.compile(optimizer=Adam(lr=cfg.LEARNING_RATE), loss='mse', metrics=[])

    BATCH_SIZE= cfg.BATCH_SIZE
    
    samples_per_epoch = sample_count((len(train)*3), BATCH_SIZE)
    nb_val_samples = sample_count((len(validation)*3), BATCH_SIZE)

    history = model.fit_generator(batch_generator(train, BATCH_SIZE), 
                                  validation_data = batch_generator(validation, BATCH_SIZE),
                                  samples_per_epoch = samples_per_epoch, 
                                  nb_val_samples = nb_val_samples,
                                  nb_epoch = nb_epoch)    
    save_model(model)
    return (model, history)

def save_model(model):
    model_json = model.to_json()
    with open("./model.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("./model.h5")
    print("Saved model to disk")
    
def do_all(model=None, nb_epoch=cfg.NB_EPOCH):
    log = load_data()
    model, hist = fit_model(log, nb_epoch=nb_epoch, model=model)
    return model

def load_image(fn):
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img    