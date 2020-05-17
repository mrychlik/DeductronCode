print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import rychlik.deductron_pkg as ded

# from rychlik.deductron_pkg.test_deductron import *
# test_exact_model()
# test_large_model_1()
# test_comb_model_1()
# test_comb_model_2()

# from rychlik.deductron_pkg.test_metropolis_learning import *
# test_learning()

# Run Tensorflow based training
from rychlik.deductron_tf.deductron_train import *
