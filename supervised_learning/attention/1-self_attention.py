#!/usr/bin/env python3
"""
Module that contains the class SelfAttention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    calculate the attention for machine translation
    """
    def __init__(self, units):
        """
        Class constructor
        """
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Call method for the attention layer
        """
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        score = self.V(tf.nn.tanh(
            self.W(s_prev_expanded) + self.U(hidden_states)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(
            attention_weights * hidden_states, axis=1)
        return context_vector, attention_weights
