import os
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


with open("SAP_ATP_BASIC.txt", "r") as f:
    document = f.read()

messages = [
    {"role": "system", "content": f"""You are an SAP ATP expert. 
Answer primarily from the document provided. If the answer is explicitly stated, use that. If you can make a reasonable inference from the document content, do so but clearly state it is an inference. If the question is completely unrelated to the document, say so.

{document}"""}
]
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )

    reply = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}\n")