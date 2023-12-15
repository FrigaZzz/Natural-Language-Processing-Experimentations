from io import open
from conllu import parse_incr

# retrieves the word
def tokenize_sentence(sentence):
    words = [];
    for token in sentence:
        words.append(token["form"]);
    return words
