# Import necessary libraries and modules
import random
import nltk
import csv
from collections import defaultdict
import math 
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

from src.log import NGramLanguageModelLogProbs

class NGramLanguageModelSmoothing(NGramLanguageModelLogProbs):
    def __init__(self, n):
        super().__init__(n)
    
    def preprocessSentence(self, sentence,test=False):
        tokens = word_tokenize(sentence)
        # delete tokens that are not in the vocabulary, and replace them with <unk>
        if(test):
            tokens = [token if token in self.ngrams.keys()  else '<unk>' for token in tokens]
        return tokens
        
    def train(self, corpus):
        # Preprocess corpus to add start and end tokens
        # + PADDING of the sentence -> <s> and </s> are added two markers that delimit the sentences
        processed_corpus = []
        for sentence in corpus:
            
            tokenized_sentence = self.preprocessSentence(sentence)
            # add padding
            processed_sentence = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']
            processed_corpus.extend(processed_sentence)

        #Extract the vocabulary
        self.word_tokens = list(set(processed_corpus))
        self.word_tokens.append('<unk>')    
        
        # Generate n-grams from the corpus
        for i in range(len(processed_corpus) - self.n + 1):
            prev_words = tuple(processed_corpus[i:i+self.n-1])
            current_word = processed_corpus[i+self.n-1]

            self.ngrams[prev_words][current_word]['frequency'] += 1
            self.ngrams[prev_words][current_word]['probability'] += 1
    
        #add the <unk> token to the vocabulary and dont initialize it
        self.ngrams[('<unk>',)]= {}
        # Apply Laplace smoothing and normalize counts to estimate probabilities
        vocabulary_size = len (self.ngrams.keys())  # Size of the vocabulary, included the padding
        for prev_words, next_words in self.ngrams.items():
            total_count = sum(next_words[word]['frequency'] for word in next_words)
            for word in next_words:
                word_count = next_words[word]['frequency']
                smoothed_count = word_count + 1  # Apply Laplace smoothing
                smoothed_probability = smoothed_count / (total_count + vocabulary_size)
                self.ngrams[prev_words][word]['probability'] = smoothed_probability
                self.ngrams[prev_words][word]['frequency'] += 1

    
            remaining_words = set(self.word_tokens) - set(next_words.keys()) 
            for word in remaining_words:
                smoothed_probability = 1 / (total_count + vocabulary_size)
                if(self.ngrams[prev_words].get(word) is None):  
                    self.ngrams[prev_words][word] = defaultdict(lambda: {"frequency": 0, "probability": 0.0})
               
                self.ngrams[prev_words][word]['frequency'] =1 
                self.ngrams[prev_words][word]['probability'] =smoothed_probability
    

            
   
   
   
# Create an instance of the n-gram language model with n=3 (trigram model)
model = NGramLanguageModelSmoothing(3)

# Train the model on a corpus (a list of sentences or words)
corpus =[
    'Peter Piper picked a peck of pickled pepper. ',
    "Where's the pickled pepper that Peter Piper picked?",
]
print(corpus)

model.train(corpus)
sentence = "Peter Piper picked a peck of pickled pepper."
sentence2 = "xxzx231 1321 Piper pepper sadasdada.£- adas"

print("Sentence Probability:", model.sentence_probability(sentence))
print("Sentence Probability:", model.sentence_probability(sentence2))

print("Input 1 Sentence Perplexity:", model.perplexity(sentence))
print("Input 2 Sentence Perplexity:", model.perplexity(sentence2))

# Generate new text using the model
seed = None
generated_text = model.generate(seed, max_length=15)
print('Generated Text:', (generated_text))

#model.printDataframe()