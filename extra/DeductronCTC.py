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
    

    def call(self, inputs, targets):
        outputs = self.deductron(inputs)
        ctc_loss = tf.nn.ctc_loss(labels = targets, logits = outputs,
                                  logit_length = targets.shape[1],
                                  label_length = 100,
                                  blank_index = -1)
        self.add_loss(ctc_loss)
        


if __name__ == '__main__':
    #
    # Sliding windows for string 'XO'
    # surrounded by blanks
    #
    tiny_inputs = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0]
    ],dtype='float32')

    tiny_targets = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0],
    ],dtype='float32')

    inputs = tiny_inputs
    targets = tiny_targets
        
    ded = DeductronCTC(n_memory = 3, input_len = 6, n_classes = 3)
    outputs = ded(inputs, targets)
        
