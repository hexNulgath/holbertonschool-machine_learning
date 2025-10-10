#!/usr/bin/env python
"""Semantic Search using BERT model."""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(corpus_path):
    """
    Perform question answering using semantic search to find the most relevant document.
    """
    # Load the QA model
    qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    qa_model = hub.load("https://www.kaggle.com/models/seesee/bert/TensorFlow2/uncased-tf2-qa/1")

    # Load the semantic search model
    semantic_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Read all documents from the corpus
    documents = []
    filenames = []
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        if not filename.lower().endswith(('.txt', '.md', '.text')):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
            filenames.append(filename)

    while True:
        sentence = question()
        if sentence is None:
            break
        else:
            most_similar_doc_content = semantic_search(documents, sentence, semantic_model)
            if most_similar_doc_content is None:
                print("A: Sorry, I do not find your question.")
            else:
                result = qa(sentence, most_similar_doc_content, qa_tokenizer, qa_model)
                if result is None:
                    print("A: Sorry, I do not understand your question.")
                else:
                    print(f"A: {result}")


def semantic_search(documents, sentence, model):
    """
    This function performs semantic search on a corpus of reference documents using a pre-trained model.
    Args:
        documents (list): A list of strings, where each string is a document from the corpus.
        sentence (str): The query sentence.
        model: The loaded Universal Sentence Encoder model.
    Returns:
        str: The content of the document most similar to the query sentence, or None if no documents are provided.
    """
    if not documents:
        return None

    corpus_embeddings = model(documents)
    query_embedding = model([sentence])[0]
    distances = np.inner(query_embedding, corpus_embeddings)
    closest_idx = np.argmax(distances)

    return documents[closest_idx]


def question():
    """Run a question-answering loop."""
    exit = ['exit', 'quit', 'goodbye', 'bye']
    x = input("Q: ")
    x = x.lower()
    if x in exit:
        print("A: Goodbye")
        return None
    else:
        return x


def qa(question, reference, tokenizer, model):
    """Answer a question based on the reference text."""
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))

    max_seq_length = 512
    if len(input_word_ids[0]) > max_seq_length:
        input_word_ids = tf.expand_dims(input_word_ids[0][:max_seq_length], 0)
        input_mask = tf.expand_dims(input_mask[0][:max_seq_length], 0)
        input_type_ids = tf.expand_dims(input_type_ids[0][:max_seq_length], 0)

    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[short_start: short_end + 1]
    if (len(answer_tokens) < 1) or (len(answer_tokens) > len(tokens) - len(question_tokens) - 3):
        return None

    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer


if __name__ == "__main__":
    question_answer("ZendeskArticles")
