#!/usr/bin/env python3
"""
Module that contains the class RNNDecoder
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Call method for the RNN Decoder
        """
        x = self.embedding(x)
        context, _ = SelfAttention(self.units)(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)
        y = self.F(y)
        
        return y, s
