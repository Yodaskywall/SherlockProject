import os
import pickle
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

whole_text = ""

for dr in os.listdir("data/books"):
    with open("data/books/" + dr, encoding="utf-8") as file:
        text = file.read()
        whole_text += text

with open("data/allbooks.pickle", "wb") as file:
    pickle.dump(encoding.encode(whole_text), file, protocol=pickle.DEFAULT_PROTOCOL)