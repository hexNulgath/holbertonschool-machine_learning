#!/usr/bin/env python3
"""
Module that contains the function uni_bleu
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    """
    s_count = {}
    for word in sentence:
        if word in s_count:
            s_count[word] += 1
        else:
            s_count[word] = 1

    r_count = {}
    for reference in references:
        for word in set(reference):
            if word in r_count:
                r_count[word] += 1
            else:
                r_count[word] = 1
            r_count[word] = max(r_count[word], reference.count(word))

    c_counts = {
        word:
        min(count, r_count.get(word, 0)) for word, count in s_count.items()}
    precision = sum(c_counts.values()) / sum(s_count.values())

    if len(sentence) > min(len(ref) for ref in references):
        bp = 1
    else:
        bp = np.exp(1 - min(len(ref) for ref in references) / len(sentence))
    precision *= bp
    return precision
