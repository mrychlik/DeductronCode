from BromelloBuilder import *
from img_to_video import *

bb = BromelloBuilder();
(unigrams,_) = bb.build_unigrams();
six = unigrams[20]
window = 3
vid = img_to_video(six,window)
play_video(vid)


