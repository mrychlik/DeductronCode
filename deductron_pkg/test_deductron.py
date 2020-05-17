from . data import *
from . WLangDecoderExact import *
from . WLangDecoderLargeModel1 import *
from . WLangDecoderCombModel1 import *
from . WLangDecoderCombModel2 import *


    # Choose one of the neural nets
    #nn = WLangDecoderExact()        # Definitely always works
    #nn = WLangDecoderLargeModel1()   # Only works on large sample
    #nn = WLangDecoderCombModel1()    # Works except for tiny sample
    #nn = WLangDecoderCombModel2()    # Works for all

n_digits = 4
def _testfun(nn, inputs, targets, descr):
    print(descr, nn(inputs).loss(targets).round(n_digits))

def test_comb_model_1():
    nn = WLangDecoderCombModel1()    # Works for all
    _testfun(nn, small_inputs, small_targets, "Small sample loss:   ");
    _testfun(nn, large_inputs, large_targets, "Large sample loss:   ");
    _testfun(nn, comb_inputs, comb_targets, "Large sample loss:   ");
    assert 0


def test_comb_model_2():
    nn = WLangDecoderCombModel2()    # Works for all
    _testfun(nn, tiny_inputs, tiny_targets, "Tiny sample loss:   ");
    _testfun(nn, small_inputs, small_targets, "Small sample loss:   ");
    _testfun(nn, large_inputs, large_targets, "Large sample loss:   ");
    _testfun(nn, comb_inputs, comb_targets, "Large sample loss:   ");
    assert 0

def test_comb_model_2():
    nn = WLangDecoderLargeModel1()    # Works for all
    _testfun(nn, tiny_inputs, tiny_targets, "Tiny sample loss:   ");
    _testfun(nn, small_inputs, small_targets, "Small sample loss:   ");
    _testfun(nn, large_inputs, large_targets, "Large sample loss:   ");
    _testfun(nn, comb_inputs, comb_targets, "Large sample loss:   ");
    assert 0

