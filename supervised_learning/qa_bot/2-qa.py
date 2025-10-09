#!/usr/bin/env python
"""Question Answering Bot using BERT model."""
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer


def answer_loop(reference):
    """Answer a question based on the reference text."""
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://www.kaggle.com/models/seesee/bert/TensorFlow2/uncased-tf2-qa/1")
    loop = True
    while loop:
        q = question()
        if q is None:
            loop = False
        else:
            a = question_answer(q, reference, tokenizer, model)
            if a is None:
                print("A: Sorry, I do not understand your question.")
            else:
                print(f"A: {a}")


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


def question_answer(question, reference, tokenizer, model):
    """Answer a question based on the reference text."""
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    if (len(answer_tokens) < 3) or (len(answer_tokens) > 50):
        return None
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
