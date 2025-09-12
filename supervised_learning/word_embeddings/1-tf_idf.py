#!/usr/bin/env python3
"""
creates a TF-IDF embedding matrix
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding matrix
    """
    bag_of_words = __import__('0-bag_of_words').bag_of_words

    # Get raw term counts and the actual features used
    counts, features = bag_of_words(sentences, vocab)

    n_docs = counts.shape[0]
    df = (counts > 0).sum(axis=0)

    idf = 1.0 + np.log((1.0 + n_docs) / (1.0 + df))

    # Apply TF-IDF: broadcast idf over rows
    tf_idf_matrix = counts * idf

    # L2 normalize rows
    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    # Avoid division by zero: leave all-zero rows as zero
    tf_idf_matrix = np.divide(
        tf_idf_matrix,
        norms,
        out=np.zeros_like(tf_idf_matrix, dtype=float),
        where=norms != 0,
    )

    return tf_idf_matrix, features
