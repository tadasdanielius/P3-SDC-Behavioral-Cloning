import cv2
import numpy as np
import math

import app.config as cfg

# Crop image. Leave only road
def crop(img):
    shape = img.shape
    img = img[math.floor(shape[0]/cfg.IMG_CROP_PARAM):shape[0]-cfg.IMG_CROP_BOTTOM, 0:shape[1]]
    return img

# Slightly make image bigger
resize = lambda x: cv2.resize(x, cfg.IMG_RESIZE_SHAPE, interpolation=cv2.INTER_AREA)

# Blur the image so sharp edges are removed
blur = lambda x: cv2.filter2D(x,-1,np.ones((5,5),np.float32)/25)

# Flip camera so we will have extra images
flip = lambda x, y: (cv2.flip(x, 1), -1*y)

# Flip camera or leave original
camera_flip = [lambda x, y: (x, y), flip]

# Prepare image:
#  crop, resize
def fix_image(img):
    img = crop(img)
    img = resize(img)
    return img

# Augmentation to generate extra images
# Slightly adjust colours
def augment_colours (img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2] * (cfg.BRIGHTNESS_STRENGHT + np.random.uniform())
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

# Shift image left/right up/down
def augment_shift_image(img, angle):
    x = cfg.WIDTH_SHIFT_RANGE * np.random.uniform() - cfg.WIDTH_SHIFT_RANGE / 2
    y = cfg.HEIGHT_SHIFT_RANGE * np.random.uniform() - cfg.HEIGHT_SHIFT_RANGE / 2

    img = cv2.warpAffine(img, np.float32([[1, 0,x], [0, 1, y]]), (img.shape[1], img.shape[0]))
    angle = (angle + x / cfg.WIDTH_SHIFT_RANGE * 2 * .2)
    return (img, angle)

def augment_image(img, angle):

    # Randomly move image up/down and left/right
    img, angle = augment_shift_image(img, angle)

    # We can flip camera and change angle*(-1), so we can double images
    # Randomly either flip or leave original
    img, angle = camera_flip[np.random.randint(len(camera_flip))](img, angle)

    # Change brightness of the image
    img = augment_colours(img)

    # Finally fix the image (crop, resize)
    img = fix_image(img)
    
    return (img, angle)
