from . data import small_inputs, small_targets
from . QuantizedDeductron import QuantizedDeductron

def _annealing_test(inputs, targets):
    net, loss = QuantizedDeductron.train(3, inputs, targets, beta_incr = 0.00001)
    print(str(net))
    return (net, loss)

def test_learning():
    _annealing_test(small_inputs, small_targets)

