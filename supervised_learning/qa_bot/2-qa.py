#!/usr/bin/env python3
"""
Answer Questions
"""


import tensorflow as tf
import tensorflow_hub as tfhub
from transformers import BertTokenizer
finishers = ["exit", "quit", "goodbye", "bye"]
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
  while True:
      text_in = input("Q: ").strip().lower()
      print(end='A: ')
      if text_in in finishers:
          print("Goodbye")
          break
      answer = question_answer(text_in, reference)
      if answer is None:
          answer = 'Sorry, I do not understand your question.'
      print(answer)
