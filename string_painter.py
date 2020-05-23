from PIL import Image, ImageDraw, ImageFont

font_dir = "./fonts"
font_size = 20
font = ImageFont.truetype(os.path.join(font_dir, "bromello-Regular.ttf"),font_size)
image = Image.new(mode='L', size=(600, 100), color='white')
draw = ImageDraw.Draw(im = image)
draw.fontmode = "1"
text='Professor Rychlik can Bromello on occasions'
draw.text(xy=(20, 40), text=text, fill='#000000', font = font)
image.save('my_test.png', 'PNG')
