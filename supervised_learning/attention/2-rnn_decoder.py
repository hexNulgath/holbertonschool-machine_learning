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
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Call method for the RNN Decoder
        """
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, _ = attention(s_prev, hidden_states)
        x = self.embedding(x)
        concat = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, s = self.gru(concat)
        y = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        return self.F(y), s
