#!/usr/bin/env python3
"""
    QA loop
"""

finishers = ["exit", "quit", "goodbye", "bye"]

while True:
    text_in = input("Q: ").strip().lower()
    print(end='A: ')
    if text_in in finishers:
        print("Goodbye")
        break
    print()
