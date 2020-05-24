from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pylab
from bbox import bbox;
import os

font_dir = "./fonts"
font_size = 20
font = ImageFont.truetype(os.path.join(font_dir, "bromello-Regular.ttf"),font_size)
image = Image.new(mode='L', size=(600, 100), color='white')
draw = ImageDraw.Draw(im = image)
draw.fontmode = "1"
text='Professor Rychlik can Bromello on occasions'
draw.text(xy=(20, 40), text=text, fill='#000000', font = font)
pixels = 255-np.asarray(image)
box = bbox(pixels);
draw.rectangle(box);
image.save('my_test.png', 'PNG')
