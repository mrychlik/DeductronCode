import numpy as np
import time

def img_to_video(x, window):
    z = np.zeros([x.shape[0],window])
    # Pad with zeros on both ends
    nrows = x.shape[0]
    ncols = x.shape[1]
    x = np.hstack([z,x,z])
    lst = 2*window*[None]
    for j in range(ncols-window):
        lst[j] = x[:,j:(j+window)]
    return lst



def play_video(lst, delay=1):
    for j in range(len(lst)):
        pylab.imshow(lst[j])
        pylab.show()
