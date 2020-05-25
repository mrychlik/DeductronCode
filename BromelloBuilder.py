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
    SpecialWidth = None
    SpecialBox = None

    def __init__(self):
        self.SpecialImage, _ = sp.draw_string(self.SpecialString)
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
        # Add special string on both ends
        (aug_im, _) = sp.draw_string(self.SpecialString + text + self.SpecialString)
        # Eliminate pixels of special string
        im = aug_im[:,(self.SpecialWidth+1):(aug_im.shape[1]-self.SpecialWidth)]
        return im

    def draw_strings(self, strings):
        nstr = len(strings)
        im = nstr * [None]
        for j in range(0, nstr):
            im[j] = self.draw_string(strings[j])
        return im
            

if __name__ == '__main__':
    bb = BromelloBuilder();
    text = 'Professor Rychlik can Bromello on occasions'
    pixels = bb.draw_string(text)
    pylab.imshow(pixels)
    pylab.show()
    
