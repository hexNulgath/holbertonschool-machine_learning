#!/usr/bin/env python3
import string
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix
    """
    s_words = {'is', 'was', 'has', 'this', 'his', 'hers', 'its', 'us'}
    punctuation = set(string.punctuation)
    # Create vocabulary if not provided
    if vocab is None:
        vocab = set()

        # First pass: collect all words as they appear
        for sentence in sentences:
            for word in sentence.split():
                word = word.lower()
                word = ''.join(
                    char for char in word if char not in punctuation)
                vocab.add(word)

        # Convert to list for processing
        vocab_list = list(vocab)

        # Second pass: remove plurals only if singular exists
        final_vocab = set()
        for word in vocab_list:
            # Check if word ends with 's' and might be a plural
            if (word.endswith('s') and
                len(word) > 1 and  # Don't process single letter words
                not word.endswith('ss') and  # Words like 'class', 'glass'
                not word.endswith('us') and  # Words like 'bus', 'plus'
                    word not in s_words):  # Common exceptions

                singular = word[:-1]
                # Only remove plural if singular form exists in vocabulary
                if singular in vocab_list:
                    final_vocab.add(singular)
                else:
                    final_vocab.add(word)
            else:
                final_vocab.add(word)

        vocab = sorted(final_vocab)

    word2idx = {word: i for i, word in enumerate(vocab)}
    embedding = [[0 for _ in range(len(vocab))] for _ in range(len(sentences))]

    for i, sentence in enumerate(sentences):
        sentence = sentence.lower()
        sentence = ''.join(
            char for char in sentence if char not in punctuation)
        for word in sentence.split():
            processed_word = word.lower()
            processed_word = ''.join(
                char for char in processed_word if char not in punctuation)

            # Apply the same plural removal logic as during vocabulary creation
            if (processed_word.endswith('s') and
                len(processed_word) > 1 and
                not processed_word.endswith('ss') and
                not processed_word.endswith('us') and
                    processed_word not in s_words):

                singular = processed_word[:-1]
                if singular in word2idx:
                    processed_word = singular

            if processed_word in word2idx:
                embedding[i][word2idx[processed_word]] += 1

    embedding = np.array(embedding)
    vocab = np.array(vocab)
    return embedding, vocab
