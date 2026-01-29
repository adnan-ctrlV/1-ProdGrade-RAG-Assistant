# This file runs the entire system

from llm import ask_llm

with open("data/docs.txt", "r") as f:
    context = f.read()

question = input("Ask a question: ")

answer = ask_llm(question=question, context=context)

print("\nAnswer: ")
print(answer)
