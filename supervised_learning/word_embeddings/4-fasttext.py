#!/usr/bin/env python3
"""
creates, builds and trains a genism fastText model
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    creates, builds and trains a genism fastText model
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
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

    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
