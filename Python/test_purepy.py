from . data import small_inputs, small_targets
from . annealing import annealing_test

def test_answer():
    annealing_test(small_inputs, small_targets)
    assert 0 == 0
