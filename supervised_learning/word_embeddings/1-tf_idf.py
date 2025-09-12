#!/usr/bin/env python3
"""
creates a TF-IDF embedding matrix
"""
import numpy as np
import string


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding matrix
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

    # Helper function to process words consistently
    def process_word(word):
        word = word.lower()
        word = ''.join(char for char in word if char not in punctuation)

        # Apply the same plural removal logic as during vocabulary creation
        if (word.endswith('s') and
            len(word) > 1 and
            not word.endswith('ss') and
            not word.endswith('us') and
                word not in s_words):

            singular = word[:-1]
            if singular in vocab:
                return singular
        return word

    # Initialize matrices
    tf = np.zeros((len(sentences), len(vocab)))
    idf = np.zeros((1, len(vocab)))

    # Calculate TF (Term Frequency)
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        word_count = 0

        for word in words:
            processed_word = process_word(word)
            if processed_word in vocab:
                j = vocab.index(processed_word)
                tf[i][j] += 1
                word_count += 1

        # Normalize TF by the number of valid words in the sentence
        if word_count > 0:
            tf[i] /= word_count

    # Calculate IDF (Inverse Document Frequency)
    for j, word in enumerate(vocab):
        count = 0
        for sentence in sentences:
            sentence_words = [process_word(w) for w in sentence.split()]
            if word in sentence_words:
                count += 1

        if count > 0:
            idf[0][j] = np.log(len(sentences) / count)
        else:
            idf[0][j] = 0

    # Calculate TF-IDF
    tf_idf_matrix = tf * idf

    return tf_idf_matrix, np.array(vocab)
