To run the code:
```
python -m app.drive --model model.json --speed 15 --adjust 1.5
```

To train network (but requires to unzip training data)
```
python -m app.train
```

There are three arguments:

* speed - The solution will try to keep maximum speed
* adjust is used to multiply predicted steering angle value by specified value
* model is the json file name of the model


# P3 Behavioral Cloning

The goal of this project is to train a deep neural network to drive a car like a human! This is a third project for self driving car nano degree program.


My aim was to build very light weight model which could be trained on CPU without much of the training data. I have used udacity provided training data set which worked well enough for both tracks. I also used all images in the training set including from left and right camera with some adjustments of steering angle.

### Preprocessing images

To produce more training data I have experimented with different techniques to generate more images. Some of the techniques I tried is:

* Zooming / Rotating images
* Shifting images left / right and up / down

It turned out that shifting images is very useful and randomly changing brightness is useful for the second track.


### Model

I have experimented with different types of pre-trained models like VGG16, Inception etc. Even if I don't train pre-trained layers and only add extra couple of layers and train only them it takes ages! to run on CPU. Without GPU it's hard to do trial-error experiments. So I ended up using comma.ai simple model which works very well. In the end I've notice that simple model works even better than more complicated but the key element in this assignment is having good training data.


### app/config.py
In this file all tuning parameters are defined

### app/drive.py
The script receives the data from udacity simulator, predicts the steering angle and calculates the throttle based on current speed.

### app/iterator.py

Data generator should be thread safe. This script defines thread-safe iterator

### app/model.py

Definition of the deep neural network model

### app/training.py

This script takes care all training logic like data generator, fitting, saving model 

### app/transform.py

This script is responsible for all kinds of image augmentations:

* Augmenting brightness
* Shifting images
* Flipping
* Cropping
* Resizing

For futher details please see [report](https://github.com/tadasdanielius/P3-SDC-Behavioral-Cloning/blob/master/report.ipynb) ipython notebook file
