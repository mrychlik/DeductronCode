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
    print(descr + " sample loss: ", nn(inputs).loss(targets).round(n_digits))

def _test_net_template(nn, scope):
    print("Model scope: " + scope)
    _test_template(nn, tiny_inputs,  tiny_targets,  "Tiny")
    _test_template(nn, small_inputs, small_targets, "Small")
    _test_template(nn, large_inputs, large_targets, "Large")
    _test_template(nn, comb_inputs,  comb_targets,  "Combined")

def test_exact_model():
    nn = WLangDecoderExact()
    _test_net_template(nn, "Exact model, always works");


def test_large_model_1():
    nn = WLangDecoderLargeModel1()    
    _test_net_template(nn, "Only works on large sample");


def test_comb_model_1():
    nn = WLangDecoderCombModel1()    
    _test_net_template(nn, "Works except for tiny sample");


def test_comb_model_2():
    nn = WLangDecoderCombModel2()
    _test_net_template(nn, "Works for all");



