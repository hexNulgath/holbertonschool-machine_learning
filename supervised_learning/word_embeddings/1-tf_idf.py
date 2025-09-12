#!/usr/bin/env python3
"""
creates a TF-IDF embedding matrix
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding matrix

    Contract
    - Inputs:
      - sentences: list of strings
      - vocab: list of words to use as features (optional). If None, it will
        be inferred using the same preprocessing as bag_of_words.
    - Outputs:
      - embeddings: numpy.ndarray of shape (n_sentences, n_features)
      - features: numpy.ndarray of shape (n_features,) with the vocabulary in
        the same order used to build embeddings.

    Notes
    - Term Frequency (TF) uses raw counts (no per-sentence normalization).
    - Inverse Document Frequency (IDF) uses ln(N / df) without smoothing.
    - Preprocessing and vocabulary building are aligned with bag_of_words.
    """
    # Reuse bag_of_words to ensure identical preprocessing/tokenization
    # Import locally to avoid circular import issues at module load time and
    # to support unconventional filenames (e.g., "0-bag_of_words.py").
    bag_of_words = __import__('0-bag_of_words').bag_of_words

    # Get raw term counts and the actual features used
    counts, features = bag_of_words(sentences, vocab)

    n_docs = counts.shape[0]
    # Document frequency: number of sentences where term appears at least once
    df = (counts > 0).sum(axis=0)

    # Smooth IDF (scikit-learn style): 1 + ln((1 + N) / (1 + df))
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
