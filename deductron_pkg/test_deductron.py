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
def _test_template(nn, inputs, targets, descr):
    print(descr, nn(inputs).loss(targets).round(n_digits))

def _test_net_template(nn):
    _test_template(nn, tiny_inputs, tiny_targets, "Tiny sample loss:   ");
    _test_template(nn, small_inputs, small_targets, "Small sample loss:   ");
    _test_template(nn, large_inputs, large_targets, "Large sample loss:   ");
    _test_template(nn, comb_inputs, comb_targets, "Large sample loss:   ");

def test_exact_model():
    nn = WLangDecoderExact()    # Definitely always works
    _test_net_template(nn);
    assert 0


def test_large_model_1():
    nn = WLangDecoderLargeModel1()    # Only works on large sample
    _test_net_template(nn);
    assert 0


def test_comb_model_1():
    nn = WLangDecoderCombModel1()    # Works except for tiny sample
    _test_net_template(nn);
    assert 0


def test_comb_model_2():
    nn = WLangDecoderCombModel2()    # Works for all
    _test_net_template(nn);
    assert 0



