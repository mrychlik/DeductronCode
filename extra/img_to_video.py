import numpy as np
import time
import matplotlib.pyplot as plt

def img_to_video(x, window):
    z = np.zeros([x.shape[0],window])
    # Pad with zeros on both ends
    nrows = x.shape[0]
    ncols = x.shape[1]
    x = np.hstack([z,x,z])
    lst = (ncols+window)*[None]
    for j in range(ncols+window):
        lst[j] = x[:,j:(j+window)]
    return lst



def play_video(lst, delay=1):
    n = len(lst)
    #nr = np.ceil(np.sqrt(n+1))
    nr = 1
    nc = n
    for j in range(n):
        plt.subplot(nr,nc,j+1)
        plt.imshow(lst[j])
    plt.show()
