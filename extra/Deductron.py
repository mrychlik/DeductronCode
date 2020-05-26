import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras import layers

class Deductron(layers.Layer):

    def __init__(self, n_memory = 3, input_len = 6, output_len = 2):
        super(Deductron, self).__init__()

        self.n_memory = n_memory
        self.input_len = input_len
        self.output_len = output_len

        W1_init = tf.random_normal_initializer()
        self.W1 = tf.Variable(initial_value = W1_init(shape = (2*n_memory,input_len),
                                                      dtype='float32'),
                              trainable = True);
                              
        B1_init = tf.random_normal_initializer()
        self.B1 = tf.Variable(initial_value = B1_init(shape = (2*n_memory,1),
                                                      dtype = 'float32'),
                              trainable = True);
        W2_init = tf.random_normal_initializer()
        self.W2 = tf.Variable(initial_value = W2_init(shape = (output_len,n_memory),
                                                      dtype='float32'),
                              trainable = True);
        
        B2_init = tf.random_normal_initializer()
        self.B2 = tf.Variable(initial_value = B2_init(shape = (output_len,1),
                                                      dtype = 'float32'),
                              trainable = True);
                                                      
    def call(self, inputs):
        n_frames = inputs.shape[0]
        h = tf.sigmoid( tf.matmul(self.W1,inputs) + self.B1 )
        [left,right] = tf.split(h, num_or_size_splits=2, axis=0)
        prod = tf.multiply(left, right) #Hadamard product
        prod = tf.unstack(prod, axis=1)
        left = tf.unstack(left, axis=1)
        u = tf.zeros([self.n_memory])
        zlst = [u]
        for t in range(1, n_frames):
            ## Memory
            u = prod[t-1] * u + ( 1.0 - left[t-1])
            zlst.append(u)
            z = tf.stack(zlst, axis = 1)
        out = 1.0 - tf.sigmoid( tf.matmul(self.W2, z) + self.B2)
        return out

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
    ],dtype='float32').transpose()

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
    ],dtype='float32').transpose()

    inputs = tiny_inputs
    targets = tiny_targets
        
    ded = Deductron(n_memory = 3, input_len = 6, output_len = 2)
    outputs = ded(inputs)
