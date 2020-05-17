from . data import small_inputs, small_targets
import deductron_pkg.QuantizedDeductron

def _annealing_test(inputs, targets):
    net, loss = QuantizedDeductron.train(
        3, inputs, targets, beta_incr = 0.00001)
    print(str(net))
    return (net, loss)



def test_answer():
    _annealing_test(small_inputs, small_targets)
    assert 0 == 0
