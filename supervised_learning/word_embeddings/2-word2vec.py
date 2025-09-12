#!/usr/bin/env python3
"""
Train and return a gensim Word2Vec model.
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    Args:
        sentences: Iterable of tokenized sentences (list[list[str]]).
        vector_size: Dimensionality of the word vectors.
        min_count: Ignores all words with total frequency lower than this.
        window: Maximum distance between current and predicted word within
        a sentence.
        negative: If > 0, negative sampling will be used; the int for negative
        specifies how many "noise words" should be drawn.
        cbow: If True, use CBOW (sg=0); if False, use Skip-gram (sg=1).
        epochs: Number of iterations (epochs) over the corpus.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        workers=workers,
        seed=seed,
        epochs=epochs,
    )

    return model
