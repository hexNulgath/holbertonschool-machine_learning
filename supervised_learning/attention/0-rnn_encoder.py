#!/usr/bin/env python3
"""
RNN Encoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    
    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN
        """
        return tf.zeros((self.batch, self.units))
    
    def call(self, x, initial):
        """
        Call method for the encoder
        """
        x = self.embedding(x)
        output, hidden = self.gru(x, initial_state=initial)
        return output, hidden