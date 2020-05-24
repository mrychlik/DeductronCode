import numpy as np

def bbox(arr):
    '''Construct bounding box of a BW image given as a pixel array.'''
    indw = np.nonzero(arr.any(axis=0))[0] # indices of non empty columns 
    indh = np.nonzero(arr.any(axis=1))[0] # indices of non empty rows
    return (indw[0],indh[0],indw[-1],indh[-1])
