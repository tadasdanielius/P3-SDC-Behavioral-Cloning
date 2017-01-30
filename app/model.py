from keras.layers import Convolution2D
from keras.models import Sequential
from keras.layers import Dropout, Flatten, ELU, Lambda, Dense
from keras.optimizers import Adam
import math
import keras

import app.config as cfg

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(cfg.IMG_RESIZE_SHAPE[0], cfg.IMG_RESIZE_SHAPE[1], 3)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv1'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv2'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv3'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, activation='elu', name='FC1'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, name='output'))
    return model