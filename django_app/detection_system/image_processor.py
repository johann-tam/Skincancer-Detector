"""
Original author(s): Johann Tammen
Modified by: Linus Ivarsson

File purpose: Processes a jpg image into an array of 784 greyscale values
"""

from PIL import Image
from django.conf import settings
import numpy as np
import tensorflow as tf
from .variables import *


# Takes a jpg image as an input, converts it to greyscale and resizes it into 28*28 pixels. Parameter filename: just
# the filename, no path. Output: Array of 784 greyscale values.
def process_jpg(filename):
    # load image and convert it to greyscale
    img = Image.open(settings.IMAGE_ROOT + '/' + filename).convert('L')
    # resize the image to the specified width and height that the dl_model was trained on
    resized_img = img.resize((IMAGEWIDTH, IMAGEHEIGHT))
    # get the pixel values for the image
    image_sequence = resized_img.getdata()
    # store the pixel values in an numpy array for the dl_model to use
    image_array = np.array(image_sequence)

    return image_array


# Transforms the array of greyscale values into a format the the CNN can accept. Input: Array of 784 greyscale values.
# Output: array with shape (amount of rows ,* (28, 28, 1))
def transform_pixel_array(pixelarray):
    image_array = np.array(pixelarray, dtype=DTYPE)
    # HSV-colorspace is in the range of 0-255, dividing by 255 results in a number between 0-1
    image_array = image_array / CSVRANGE
    # Reshaping the image to fit the dimensions expected by the CNN
    image_array = image_array.reshape(1, *(IMAGEWIDTH, IMAGEHEIGHT))
    # add padding to picture from 28x28 to 32x32
    image_array = tf.pad(tensor=image_array, paddings=[[0, 0], [PADDING, PADDING], [PADDING, PADDING]])
    # add new axis corresponding to the required channels for the DenseNet121 model
    image_array = np.repeat(image_array[..., np.newaxis], CHANNELS, CHANNELPOSITION)
    return image_array
