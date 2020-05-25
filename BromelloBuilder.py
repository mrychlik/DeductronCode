import string_painter as sp
import pylab

class BromelloBuilder:
    Alphabet = '!"#$%&''()*+,-./' \
        '0123456789' \
        ':;<=>?@' \
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
        '[\]^abcdefghijklmnopqrstuvwxyz' \
        '{|}~ '
    FontSize = 48  
    SpecialString = '[kY]'  
    SpecialImage = None
    SpecialBox = None
    SpecialWidth = None

    def __init__(self):
        (self.SpecialImage, self.SpecialBox) = sp.draw_string(self.SpecialString)
        self.SpecialWidth = self.SpecialImage.shape[1]


    def __str__(self):
        return ("{}:\n"
                "Alphabet: {}\n"
                "FontSize: {}\n"
                "SpecialString: {}\n").format(self.__class__.__name__,
                                              self.Alphabet,
                                              self.FontSize,
                                              self.SpecialString)
    def draw_string(self, text):
        (aug_im, box) = sp.draw_string(self.SpecialString + text + self.SpecialString)
        im = aug_im[:,self.SpecialWidth:(aug_im.shape[1]-self.SpecialWidth)]
        return (im, box)



if __name__ == '__main__':
    bb = BromelloBuilder();
    pixels, box, = bb.draw_string('abc')
    pylab.imshow(pixels)
    pylab.show()
    
