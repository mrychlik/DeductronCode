#----------------------------------------------------------------
# File:     test_bromello_builder.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:37:15 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------

from BromelloBuilder import *
from img_to_video import *

bb = BromelloBuilder();
unigrams,_ = bb.build_unigrams();
six = unigrams[20]
window = 3
vid = img_to_video(six,window)
play_video(vid)


