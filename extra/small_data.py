#----------------------------------------------------------------
# File:     small_data.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:59:07 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------
# @brief    Small data
import numpy as np

# Training image
_small_image = [
    [0,   0,   0],     
    [0,   0,   1],
    [0,   1,   0],
    [1,   0,   0],
    [0,   1,   0],
    [0,   0,   1],
    [1,   0,   0],
    [0,   1,   0],
    [0,   0,   1],
    [0,   1,   0],
    [1,   0,   0],
    [0,   1,   0],
    [0,   0,   1],
    [0,   1,   0],
    [1,   0,   0],
    [0,   0,   1],
    [0,   1,   0],
    [1,   0,   0],
    [0,   1,   0],
    [0,   0,   1],
    [0,   1,   0],
    [1,   0,   0],
    [0,   1,   0],
    [0,   0,   1],
    [1,   0,   0],
    [0,   1,   0],
    [0,   0,   1],
    [0,   1,   0],
    [1,   0,   0],
    [0,   0,   0],
    ]
    
## Target outputs
_small_targets = [
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 1],
    [0, 0],
    [0, 0],
    ]

def img_to_input(x):
    """Converts W-language image x to a list of 2-column windows."""
    y = list(x);y.pop(0)                   #Shift x left 
    z = list(x);z.pop(-1)                  #Shift x right
    # Combine
    return np.array(list(map(lambda x: x[1]+x[0] , zip(y, z))),dtype='float32')

small_inputs = img_to_input(_small_image)
small_targets = np.array(_small_targets)
small_labels = 'XOOXXO'
