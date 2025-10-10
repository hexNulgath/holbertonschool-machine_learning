#!/usr/bin/env python
"""Semantic Search using BERT model."""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
qa = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Perform question answering using semantic search to find the most relevant document.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://www.kaggle.com/models/seesee/bert/TensorFlow2/uncased-tf2-qa/1")

    exit_commands = {"exit", "quit", "goodbye", "bye"}

    # Read all documents from the corpus
    documents = []
    filenames = []
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
            filenames.append(filename)
    while True:
        sentence = question()
        if sentence is None:
            break
        else:
            most_similar_idx = semantic_search(documents, sentence, tokenizer, model, filenames)
            if most_similar_idx is None:
                print("A: Sorry, I do not understand your question.")
            else:
                result = qa(documents[most_similar_idx], sentence)
                if result is None:
                    print("A: Sorry, I do not understand your question.")
                else:
                    print(f"A: {result}")

def semantic_search(documents, sentence, tokenizer, model, filenames):
    """
    Perform semantic search to find the most relevant answer from the corpus.
    """
    # Store similarity scores for each document
    similarity_scores = []

    for doc_content in documents:
        # Tokenize question and current document
        question_tokens = tokenizer.tokenize(sentence)
        paragraph_tokens = tokenizer.tokenize(doc_content)

        # Check if the total length exceeds model's maximum (512 tokens)
        max_len = 512
        if len(question_tokens) + len(paragraph_tokens) + 3 > max_len:
            # Truncate paragraph tokens to fit within model limits
            paragraph_max_len = max_len - len(question_tokens) - 3
            paragraph_tokens = paragraph_tokens[:paragraph_max_len]

        # Prepare input tokens and segments
        tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_word_ids)
        input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

        # Convert to tensors
        input_word_ids, input_mask, input_type_ids = map(
            lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_type_ids)
        )

        # Get model predictions
        outputs = model([input_word_ids, input_mask, input_type_ids])

        # Use the start logits as a similarity score
        # Higher max start logit indicates better match
        max_start_logit = tf.reduce_max(outputs[0]).numpy()
        similarity_scores.append(max_start_logit)

    # Find the document with highest similarity score
    if not similarity_scores:
        return None

    most_similar_idx = np.argmax(similarity_scores)

    return most_similar_idx

def question():
    """Run a question-answering loop."""
    exit = ['exit', 'quit', 'goodbye', 'bye']
    x = input("Q: ")
    x = x.lower()
    if x in exit:
        print("A: Goodbye!")
        return None
    else:
        return x

if __name__ == "__main__":
    question_answer("ZendeskArticles")