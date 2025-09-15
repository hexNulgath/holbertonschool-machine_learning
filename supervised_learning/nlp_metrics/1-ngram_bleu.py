#!/usr/bin/env python3
"""Module that contains the function ngram_bleu"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence
    """
    # Get n-grams for the candidate sentence
    candidate_ngrams = list(ngram(sentence, n))
    candidate_counts = count_ngrams(candidate_ngrams)

    # Get n-grams for all references
    reference_ngrams_list = [list(ngram(ref, n)) for ref in references]

    # Calculate maximum reference counts for each n-gram
    mrc = {}
    for ref_ngrams in reference_ngrams_list:
        ref_counts = count_ngrams(ref_ngrams)
        for ngram_token, count in ref_counts.items():
            if ngram_token not in mrc or count > mrc[ngram_token]:
                # max reference count
                mrc[ngram_token] = count

    # Calculate clipped counts
    clipped_counts = {}
    for ngram_token, count in candidate_counts.items():
        clipped_counts[ngram_token] = min(
            count, mrc.get(ngram_token, 0))

    # Calculate precision
    if len(candidate_ngrams) == 0:
        precision = 0
    else:
        precision = sum(clipped_counts.values()) / len(candidate_ngrams)

    # Calculate brevity penalty
    candidate_length = len(sentence)
    closest_ref_length = min(len(ref) for ref in references)

    if candidate_length > closest_ref_length:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_length / candidate_length)

    # Final BLEU score
    bleu_score = bp * precision

    return bleu_score


def ngram(sentence, n):
    """function that creates n-grams from a sentence"""
    n_grams = []
    sentence_length = len(sentence)

    if n <= 0 or n > sentence_length:
        return n_grams

    for i in range(sentence_length - n + 1):
        n_gram = tuple(sentence[i:i + n])
        n_grams.append(n_gram)

    return n_grams


def count_ngrams(ngrams_list):
    """Count occurrences of each n-gram in a list"""
    counts = {}
    for ngram_token in ngrams_list:
        counts[ngram_token] = counts.get(ngram_token, 0) + 1
    return counts
