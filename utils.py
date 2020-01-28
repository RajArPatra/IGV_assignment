# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:49:12 2020

@author: kb
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Prompt for limited number of options
def promptForInputCategorical(message, options):
    """
    Prompts for user input with limited number of options (not used in this project)
    :param message: Message displayed to the user
    :param options: limited number of options. 
    Prompt will repeat until one of provided options typed correctly
    :return: user response
    """
    response = ''

    options_list = ', '.join(options)

    while response not in options:
        response = input('{} ({}): '.format(message, options_list))

    return response


def promptForInt(message):
    """
    Prompting for Integer input
    :param message: Informative message when prompting for integer input
    :return: integer input
    """
    result = None
    ######################## Your code #############################


    ######################## End of your code ######################
    return result

def promptForFloat(message):
    """
    Prompting for Float
    :param message: Informative message when prompting for float input
    :return: integer input
    """
    result = None

    while result is None:
        try:
            result = float(input(message))
        except ValueError:
            pass
    return result


def putThrs(img, low, high):
    """
    Was used at the stage when determining thresholds for binarization
    put text on image by showing lower threshold is this and uuuer threshold is this
    at co-ordinate (100, 100) use defualt color
    """


def putText(img, text, origin=(100, 100), scale=1.0, color=(255, 0, 0), thickness=2):
    """
    Wrapper for OpenCV putText()
    :param img: 
    :param text: 
    :param origin: 
    :param scale: 
    :param color: 
    :param thickness: 
    :return: 
    """
    ######################## Your code #############################


    ######################## End of your code ######################


def drawRect(img, lx, ly, rx, ry, color=(0, 255, 0), thickness=2):
    """
    Wrapper for OpenCV rectangle
    :param img: 
    :param lx: 
    :param ly: 
    :param rx: 
    :param ry: 
    :param color: 
    :param thickness: 
    :return: 
    """
    ######################## Your code #############################


    ######################## End of your code ######################    


def funcSpace(argSpace, fitParams):
    """
    Creates a space of quadratic function f(y) = ay^2 + by + c values given a space of variables
    :param argSpace: space of variables, may be a single value
    :param fitParams: it has  3 parameters a, b, c
    :return: space of function values
    """
    
    val = None
    ######################## Your code #############################


    ######################## End of your code ######################

    return val


def curvature(fitParams, variable, scale=1):
    """
    :param fitParams: 2nd order polynomial params (a, b, c in f(y) = ay^2 + by + c). Passing just a tuple of
    'a' and 'b' is enough
    :param variable: the point where curvature being evaluated (passing 'linspace' should return an array of curvatures
    for a given linspace.
    :param scale: number of units per pixel
    :return: value of curvature in units
    """
    ######################## Your code #############################


    ######################## End of your code ######################


def plot(img, figsize=(12, 12), title=None, axis='off', cmap=None):
    """
    Wrapper for matplotlib.pyplot imshow. Used for jupyter notebook
    :param img: 
    :param figsize: 
    :param title: 
    :param axis: 
    :param cmap: 
    :return: 
    """
    ######################## Your code #############################


    ######################## End of your code ######################


def timeStamp():
    '''
    :return: a time format with date month year hour minute second
    '''
    import datetime
    now = datetime.datetime.now()
    y = now.year
    d = now.day
    mo = now.month
    h = now.hour
    m = now.minute
    s = now.second

    ######################## Your code #############################


    ######################## End of your code ######################


def drawBoxes(img, bBoxes, color=(0, 255, 0), thickness=4):
    """
    Universal bounding box painter, regardless of bBoxes format 
    :param img: image of interest
    :param bBoxes: list of bounding boxes.
    :param color: 
    :param thickness: 
    :return: 
    """
    ######################## Your code #############################


    ######################## End of your code ######################


# By Vivek Yadav: https://github.com/vxy10/ImageAugmentation
def transform_image(img, ang_range, shear_range, trans_range, brightness=False):
    """
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.
    A Random uniform distribution is used to generate different parameters for transformation
    """

    # Rotation
    ######################## Your code #############################


    ######################## End of your code ######################

    # Translation
    ######################## Your code #############################


    ######################## End of your code ######################

    # Shear
    ######################## Your code #############################


    ######################## End of your code ######################

    # Brightness
    if brightness:
        img = augment_brightness(img)

    return img


def augment_brightness(image):
    
    #idea -: https://github.com/vxy10/ImageAugmentation
    hsv_img = hsv(image)

    random_brightness = np.random.uniform(0.75, 1.25)

    def clamp(a):
        return min(255, a * random_brightness)

    vfunc = np.vectorize(clamp)

    hsv_img[:, :, 2] = vfunc(hsv_img[:, :, 2])

    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


def change_colorspace(x, new_color_space, ch_to_heq=None):
    '''
    use cv2.cvtColor
    for ch_to_hq first change color space then use hist_eq defined earlier
    '''
    ######################## Your code #############################


    ######################## End of your code ######################


def hls(x, ch_to_heq=None):
    return change_colorspace(x, cv2.COLOR_RGB2HLS, ch_to_heq)


def hsv(x, ch_to_heq=None):
    return change_colorspace(x, cv2.COLOR_RGB2HSV, ch_to_heq)


def yuv(x, ch_to_hec=None):
    return change_colorspace(x, cv2.COLOR_RGB2YUV, ch_to_hec)


def hist_eq(x, ch_to_heq=None):
    """
    Histogram equalization. Equalizes each channel separately.
    :param x: 
    :param ch_to_heq: 
    :return: 
    """
    ######################## Your code #############################


    ######################## End of your code ######################


def colorHeatMap(heatMapMono, cmap=cv2.COLORMAP_HOT):
    """
    Makes an RGB version of the 1-channel heatMap
    :param heatMapMono: 
    :param cmap: The color map of choice
    :return: RGB heatMap
    
    1. use histogram equalization(cv2.equalizeHist)
    2. use cv2.applyColormap 
    3. change space
    """
    ######################## Your code #############################


    ######################## End of your code ######################


def rgbImage(imageFileName, resize=False):
    """
    Opens image as RGB with OpenCV
    :param imageFileName: 
    :param resize: Halves width and height if True and use inter area interpolation
    :return: RGB image
    """
    ######################## Your code #############################


    ######################## End of your code ######################


def flipImage(image):
    """
    Horizontal flip with OpenCV
    :param image: 
    :return: Horizontally-flipped image
    """
    ######################## Your code #############################


    ######################## End of your code ######################
