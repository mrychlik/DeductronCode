from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pylab
from bbox import bbox;
import os

DEFAULT_TEXT = 'Professor Rychlik can Bromello on occasions'
DEFAULT_FONT_DIR = "./fonts"
DEFAULT_FONT_SIZE = 20

def draw_string(text=DEFAULT_TEXT, font_dir=DEFAULT_FONT_DIR, font_size = DEFAULT_FONT_SIZE):
    font = ImageFont.truetype(os.path.join(font_dir, "bromello-Regular.ttf"),font_size)
    image = Image.new(mode='L', size=(600, 100), color='white')
    draw = ImageDraw.Draw(im = image)
    draw.fontmode = "1"

    draw.text(xy=(20, 40), text=text, fill='#000000', font = font)
    pixels = 255-np.asarray(image)
    box = bbox(pixels);
    draw.rectangle(box);
    image.save('my_test.png', 'PNG')
    return (pixels[box[1]:box[3]][box[0]:box[2]], box )

if __name__ == '__main__':
    pixels,box = draw_string(DEFAULT_TEXT)
    pylab.imshow(pixels)
    pylab.show()
