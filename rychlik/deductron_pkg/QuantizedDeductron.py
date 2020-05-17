import DeductronBase
import numpy as np
import copy
import random

class QuantizedDeductron(DeductronBase):

    ADMISSIBLE_WEIGHTS = [-1,0,1]
    ADMISSIBLE_BIASES  = [0,1,2,3,4,5]

    REP_MAX = 8192
    BETA_MAX = 10

    def __init__(self, n_in, n_memory, n_out, beta = 1):
        super(QuantizedDeductron, self).__init__(beta = beta, shift = 0.5)
        self.__create_random_weights(n_in, n_memory, n_out)

    def __create_random_weights(self, n_in, n_memory, n_out):
        self.W1 = np.reshape(
            random.choices(QuantizedDeductron.ADMISSIBLE_WEIGHTS,
                           k = n_in * 2 *n_memory), (2*n_memory, n_in))
        self.B1 = np.reshape(
            random.choices(QuantizedDeductron.ADMISSIBLE_BIASES,
                           k = 2 *n_memory), (2*n_memory, 1))        
        self.W2 = np.reshape(
            random.choices(QuantizedDeductron.ADMISSIBLE_WEIGHTS,
                           k = n_memory * n_out), (n_out, n_memory))        
        self.B2 = np.reshape(
            random.choices(
                QuantizedDeductron.ADMISSIBLE_BIASES,
                k = n_out),
            (n_out, 1))

    def _modify_W1(self):
        m,n = self.W1.shape
        i = random.choice(range(m))
        j = random.choice(range(n))
        old = self.W1[i,j]
        self.W1[i,j] = random.choice(
            QuantizedDeductron.ADMISSIBLE_WEIGHTS)

        return ('W1', (i, j), old)

    def _modify_B1(self):
        i = random.choice(range(self.B1.shape[0]))
        old = self.B1[i,0]
        self.B1[i] = random.choice(
            QuantizedDeductron.ADMISSIBLE_BIASES) - 0.5

        return ('B1', (i), old)

    def _modify_W2(self):
        m,n = self.W2.shape
        i = random.choice(range(m))
        j = random.choice(range(n))
        old = self.W2[i,j]
        self.W2[i,j] = random.choice(
            QuantizedDeductron.ADMISSIBLE_WEIGHTS)

        return ('W2', (i, j) , old)

    def _modify_B2(self):
        i = random.choice(range(self.B2.shape[0]))
        old = self.B2[i,0]
        self.B2[i] = random.choice(
            QuantizedDeductron.ADMISSIBLE_BIASES) - 0.5

        return ('B2', (i), old)

    def modify(self):
        i = random.choice(range(4))
        if i == 0:
            return self._modify_W1()
        elif i == 1:
            return self._modify_B1()            
        elif i == 2:
            return self._modify_W2()            
        elif i == 3:
            return self._modify_B2()            

    def restore(self, mod_data):
        name, idx, old = mod_data
        if name == 'W1':
            i, j = idx
            self.W1[i,j] = old
        elif name == 'B1':
            i = idx
            self.B1[i] = old
        elif name == 'W2':
            i, j = idx
            self.W2[i,j] = old
        elif name == 'B2':
            i = idx
            self.B2[i] = old

    def run_loss(self, inputs, targets):
        return self(inputs).loss(targets)
            
    @staticmethod
    def train(n_memory, inputs, targets, beta_incr = 0.001):
        n_in, _ = inputs.shape
        n_out, _ = targets.shape        

        print("**** Simulated annealing ****")
        net = QuantizedDeductron(n_in, n_memory, n_out, beta = 0)
        E0 = net.run_loss(inputs, targets)
        net_best = copy.deepcopy(net)
        E_best = E0
        print("%10s %10s %10s %10s"
              % ("Iteration", "Loss", "Best Loss", "Inv. Temp."))
        iter = 0
        rep = 0
        while net.beta < QuantizedDeductron.BETA_MAX:
            mod_data = net.modify()
            E1 = net.run_loss(inputs, targets)
            prob = np.exp( -net.beta * (E1 - E0) );
            if  random.uniform(0, 1) < prob:
                rep = 0
                E0 = E1         # Accept
                if  ( iter % 1000 == 0) or ( E0 < E_best ):
                    print("%10.3d %10.3f %10.3f %10.3f"
                          % (iter, E0, E_best, net.beta))
                    
                if E0 < E_best:
                    E_best = E0
                    net_best = copy.deepcopy(net)
            else:
                # Reject, restore weight, update
                rep += 1
                if (rep < 1000 and rep % 100 == 0) or (rep % 1000 == 0):
                    print("Repeats:", rep)

                if rep >= QuantizedDeductron.REP_MAX:
                    print ("Restart on iteration", iter,
                               "repetitions:", rep) 
                    # Restart
                    net = net_best
                    rep = 0
                else:
                    net.restore(mod_data)
            iter += 1
            net.beta += beta_incr
        return (net_best, E_best)
