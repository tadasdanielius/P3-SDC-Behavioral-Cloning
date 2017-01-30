# Resize cropped image to the following shape
IMG_RESIZE_SHAPE = (64, 64)

# Shift camera when taking left or right image
SHIFT_CAMERA = 0.25

# Extra logging
VERBOSE = False

# Brightness adjustment strenght
BRIGHTNESS_STRENGHT = 0.2

# Crpping
# Leave only 1/img_crop_param of the image
IMG_CROP_PARAM = 5.5
# Crop pixels from the bottom
IMG_CROP_BOTTOM = 25

WIDTH_SHIFT_RANGE = 100
HEIGHT_SHIFT_RANGE = 40

# Training parameters
NB_EPOCH = 8
LEARNING_RATE = 0.0001
BATCH_SIZE = 128

# For training data generator
# Every image which has steering angle within specified range by parameter PROB_RANGE
# will be considered whatever to include in the trainin batch or apply again augmentation
# The uniform random number is generated and compared to threshhold if the number is smaller
# than threshhold it will be re-augmented and the process will repeat
THRESHOLD = 0.85
PROB_RANGE = 0.08
