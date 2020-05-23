from data import small_inputs, small_targets
from deductron import QuantizedDeductron
import datetime

def _annealing_test(inputs, targets):
    net, loss = QuantizedDeductron.train(3, inputs, targets, beta_incr = 0.00001)
    print(str(net))
    return (net, loss)

def test_learning():
    _annealing_test(small_inputs, small_targets)

if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    print(run_name)
    test_learning()
