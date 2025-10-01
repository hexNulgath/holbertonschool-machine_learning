#!/usr/bin/env python3
"""
creates a dataset
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    loads and preps a dataset for machine translation
    """
    def __init__(self):
        """constructor"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train',
            as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation',
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', clean_up_tokenization_spaces=True)
        tokenizer_pt.train_new_from_iterator(pt_sentences, 2**13)
        tokenizer_en.train_new_from_iterator(en_sentences, 2**13)

        return tokenizer_pt, tokenizer_en
