#----------------------------------------------------------------
# File:     string_painter.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:38:02 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pylab
from bbox import bbox
import os

DEFAULT_TEXT = 'Professor Rychlik can Bromello on occasions'
DEFAULT_FONT_DIR = "./fonts"
DEFAULT_FONT_SIZE = 20

def _bounding_box(arr):
    '''Construct bounding box of a BW image given as a pixel array.'''
    indw = np.nonzero(arr.any(axis=0))[0] # indices of non empty columns 
    indh = np.nonzero(arr.any(axis=1))[0] # indices of non empty rows
    return (indw[0],indh[0],indw[-1]+1,indh[-1]+1)

def draw_string(text=DEFAULT_TEXT, font_dir=DEFAULT_FONT_DIR, font_size = DEFAULT_FONT_SIZE):
    font = ImageFont.truetype(os.path.join(font_dir, "bromello-Regular.ttf"),font_size)
    image = Image.new(mode='L', size=(600, 100), color='white')
    draw = ImageDraw.Draw(im = image)
    draw.fontmode = "1"
    draw.text(xy=(20, 40), text=text, fill='#000000', font = font)
    pixels = 255 - np.asarray(image)
    box = _bounding_box(pixels)
    draw.rectangle(box)
    #image.save('my_test.png', 'PNG')
    q = pixels[box[1]:box[3]]
    return ( q[:,box[0]:box[2]], box)

if __name__ == '__main__':
    (pixels,box) = draw_string(DEFAULT_TEXT)
    pylab.imshow(pixels)
    pylab.show()
