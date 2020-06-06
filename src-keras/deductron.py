'''
Implementation of the deductron recurrent neural network layer.

This work is a derivative of the LSTM implementation by the Keras team. Contributions from those contributors copyright (c) 2015-2019 those individuals.

Contributions from Dylan Murphy are copyright (c) 2020 Dylan Murphy.

This software is licensed under the MIT license.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

import keras
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

from keras.layers import Layer, RNN

_clipped_relu = lambda x: keras.activations.relu(x, max_value = 1)

class DeductronCell(Layer):
    '''
    Cell class for the Deductron layer. Although the activation may be replaced with an activation other than a logistic sigmoid, 
    any activation that takes values outside of [0, 1] complicates the interpretation of the V-gate as implementing a
    specific deductive step from propositional calculus.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use for the perceptron layers.
            (see [activations](../activations.md)).
            Default: logistic sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `input_kernel` and `output_kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs. (not currently implemented!!)
    '''
    def __init__(self, units,
                 activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 **kwargs):
        super(DeductronCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.state_size = self.units
        self.output_size = self.units
        self._dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_kernel = self.add_weight(shape=(input_dim, 2 * self.units),
                                      name='input_kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.output_kernel = self.add_weight(shape=(self.units, self.units),
                                      name='output_kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            bias_initializer = self.bias_initializer
            self.input_bias = self.add_weight(shape=2*self.units,
                                        name='input_bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.output_bias = self.add_weight(shape=self.units,
                                        name='output_bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, state, training = None):
        h = self.activation(K.dot(inputs, self.input_kernel) + self.input_bias) # Input perceptron-type cell
        u = h[:, :self.units]                                                   # Demux
        v = h[:, self.units:]
        z = _vgate(state[0], u, v)                                              # Logic gate
        h = self.activation(K.dot(z, self.output_kernel) + self.output_bias)
        return h, [z]

class Deductron(RNN):
    '''
    Layer class for the deductron recurrent neural network.
    '''
    def __init__(self, units,
                 activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if K.backend() == 'theano' and (dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = DeductronCell(units,
                        activation=activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout)
        super(Deductron, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(Deductron, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

def _ugate(z, u, v):
    return (1 - z) * u + z * v

def _vgate(z, u, v):
    return (1 - u) * (1 - v) * z + u