#!/usr/bin/env python3
"""
7. Transformer Encoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    create an encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        drop_rate - the dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        x - a tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder block
        training - a boolean to determine if the model is training
        mask - the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm)
            containing the block’s output
        """
        attn_output, _ = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + attn_output)
        dense_output = self.dense_output(out1)
        out2 = self.layernorm2(out1 + dense_output)
        return out2
