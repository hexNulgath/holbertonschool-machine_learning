#!/usr/bin/env python3
"""
converts a gensim word2vec model to a keras Embedding layer
"""
from tensorflow.keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    """
    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    weights = model.wv.vectors
    vocab_size, embedding_size = weights.shape

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=[weights],
    )

    return embedding
