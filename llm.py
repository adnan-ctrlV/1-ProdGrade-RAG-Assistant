# This file is responsible for all LLM functionality

from openai import OpenAI
from config import OPENAI_API_KEY
from prompts import system_prompt

client = OpenAI(api_key=OPENAI_API_KEY)


def ask_llm(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt()},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0
    )

    return response.choices[0].message.content


# [0] is the length to be returned. 0th Index. The text is usually one long string.
