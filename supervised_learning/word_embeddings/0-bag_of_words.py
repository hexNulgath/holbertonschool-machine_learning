#!/usr/bin/env python3
import string
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix
    """
    # Create vocabulary if not provided
    if vocab is None:
        vocab = list()
        for sentence in sentences:
            for word in sentence.split():
                word = word.lower()
                word = ''.join(char for char in word if char not in string.punctuation)
                if word.endswith("s"):
                    word = word[:-1]
                if word not in vocab:
                    vocab.append(word)
        
        vocab = sorted(vocab)

    word2idx = {word: i for i, word in enumerate(vocab)}
    embedding = [[0 for _ in range(len(vocab))] for _ in range(len(sentences))]
    for i, sentence in enumerate(sentences):
        sentence = sentence.lower()
        sentence = ''.join(char for char in sentence if char not in string.punctuation)
        for word in sentence.split():
            if word in word2idx:
                embedding[i][word2idx[word]] += 1
    embedding = np.array(embedding)
    return embedding, vocab