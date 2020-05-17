import numpy as np

class DeductronBase:  
    '''Deductron base. Does not define weights and biases.  Add suitable
    fields W1, B1, W2, B2 to make it work.

    '''
    W1 = None
    B1 = None
    W2 = None
    B2 = None
    out = None

    def __init__(self, beta = 1, shift = 0):
        ''' Sets beta and shift, which are applied to activations
        before standard (rising) sigmoid is applied. '''
        self.beta = beta
        self.shift = shift

    def sigmoid(self, x):
        ''' Implements sigmoid function. '''
        return 1 / (1 + np.exp(- self.beta * (x - self.shift)))


    def __call__(self, input):
        ''' Runs a deductron on input and returns output '''

        #
        # Run input classifier
        #
        h = self.sigmoid(  self.W1 @ input + self.B1)

        #
        # Split neuron outputs into two groups
        #
        left,right = np.split(h, 2)
        n_frames = 1 if len(input.shape) == 1  else input.shape[1]
        n_memory = left.shape[0]

        #
        # Implement V-gate
        #
        prod = left * right
        z = np.zeros((n_memory, n_frames))
        for t in range(1,n_frames):
            z[:,t] = prod[:,t-1] * z[:,t-1] + (1.0 - left[:,t-1])

        #
        # Classify memories
        #
        self.out = 1.0 - self.sigmoid( self.W2 @ z + self.B2 )

        return self

    def loss(self, targets):
        '''Loss - sum of squares of errors.'''
        diff = self.out - targets
        return np.sum(np.square(diff))

    def __str__(self):
        return ("{}:\n"
                "beta: {}\n"
                "shift: {}\n"                
                "W1:\n{}\n"
                "B1:\n{}\n"
                "W2:\n{}\n"
                "B2:\n{}\n").format(self.__class__.__name__,
                                    self.beta,
                                    self.shift,
                                    self.W1,
                                    self.B1,
                                    self.W2,
                                    self.B2)




