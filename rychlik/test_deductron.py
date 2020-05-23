from data import *
from deductron import *
import datetime

#
# We choose one of the neural nets:
#
#    WLangDecoderExact         - Definitely always works
#    WLangDecoderLargeModel1   - Only works on large sample
#    WLangDecoderCombModel1    - Works except for tiny sample
#    WLangDecoderCombModel2    - Works for all
#
# We test on 4 training datasets.

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



if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    print(run_name)
    test_exact_model()
    test_large_model_1()
    test_comb_model_1()
    test_comb_model_2()
