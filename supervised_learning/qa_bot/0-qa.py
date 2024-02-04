#!/usr/bin/env python3
"""
    QA bot, ex0
"""


import tensorflow as tf
import tensorflow_hub as tfhub
from transformers import BertTokenizer


def question_answer(question, reference):
    url = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
    model = tfhub.load(url)
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    qtok = tokenizer.tokenize(question)
    ref = tokenizer.tokenize(reference)
    itok = ['[CLS]'] + qtok + ['[SEP]'] + ref + ['[SEP]']
    input_ids = tf.constant([tokenizer.convert_tokens_to_ids(itok)])
    start, end = model(input_ids)
    st_index = tf.argmax(start, axis=1)[0].numpy().item()
    end_index = tf.argmax(end, axis=1)[0].numpy().item()
    atok = ref[st_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(atok)

    return answer
