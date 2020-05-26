#----------------------------------------------------------------
# File:     DeductronCTC.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@email.arizona.edu)
# Date:     Tue May 26 07:37:43 2020
# Copying:  (C) Marek Rychlik, 2019. All rights reserved.
# 
#----------------------------------------------------------------

import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from Deductron import Deductron


class DeductronCTC(tf.keras.Model):
    def __init__(self, name='DeductronCTC',
                 n_memory = 3,
                 input_len = 6,
                 n_classes = 2,
                 **kwargs):
        super(DeductronCTC, self).__init__(name=name, **kwargs)
        self.deductron = Deductron(n_memory=n_memory, input_len = input_len,
                                   output_len = n_classes)
    

    def call(self, inputs, labels):
        outputs = self.deductron(inputs)
        ctc_loss = tf.nn.ctc_loss(labels = labels, logits = outputs,
                                  logit_length = targets.shape[1],
                                  label_length = 100,
                                  blank_index = -1)
        self.add_loss(ctc_loss)
        


if __name__ == '__main__':
        
    ded = DeductronCTC(n_memory = 3, input_len = 6, n_classes = 3)
    outputs = ded(inputs, targets)
        
