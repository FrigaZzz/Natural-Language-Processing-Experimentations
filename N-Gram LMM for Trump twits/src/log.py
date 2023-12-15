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

from src.base import NGramLanguageModel


class NGramLanguageModelLogProbs(NGramLanguageModel):
    def __init__(self, n):
        super().__init__(n)
     
    def sentence_probability(self, sentence,debug=False):
        # Preprocess the input sentence
        # only if sentence doesnt have padding
        if(sentence[0]!='<s>' and sentence[-1]!='</s>'):
            tokenized_sentence = self.preprocessSentence(sentence)
            processed_sentence = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']
            
    
        # Initialize probability to maximum in logaritmic space
        probability = np.log(1.0)
        
        
        # Iterate over the sentence to compute the probability
        for i in range(len(processed_sentence) - self.n + 1):
            prev_words = tuple(processed_sentence[i:i+self.n-1])
            current_word = processed_sentence[i+self.n-1]

            # Check if the n-gram exists in the language model
            if prev_words in self.ngrams and current_word in self.ngrams[prev_words]:
                # Multiply the probability by the conditional probability of the current word given the previous words
                prob = self.ngrams[prev_words][current_word]['probability']
                log_prob = np.log(prob)
                probability += log_prob
                if debug:
                    print("P({}|{}) = {}".format(current_word, prev_words, prob))
            else:
                # if the n-gram doesn't exist, return 0.0 or smooth the probability
                probability +=  np.log(1e-10)
                if debug:
                    print("P({}|{}) = {}".format(current_word, prev_words, 'N/A'))

        if(debug):
            print("Probability of the sentence: {}".format(probability))
        return probability
    
    def sentence_probability(self, sentence,debug=False):
        # Preprocess the input sentence
        # only if sentence doesnt have padding
        if(sentence[0]!='<s>' and sentence[-1]!='</s>'):
            tokenized_sentence = self.preprocessSentence(sentence)
            processed_sentence = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']
            
    
        # Initialize probability to maximum in logaritmic space
        probability = np.log(1.0)
        
        
        # Iterate over the sentence to compute the probability
        for i in range(len(processed_sentence) - self.n + 1):
            prev_words = tuple(processed_sentence[i:i+self.n-1])
            current_word = processed_sentence[i+self.n-1]

            # Check if the n-gram exists in the language model
            if prev_words in self.ngrams and current_word in self.ngrams[prev_words]:
                # Multiply the probability by the conditional probability of the current word given the previous words
                prob = self.ngrams[prev_words][current_word]['probability']
                log_prob = np.log(prob)
                probability += log_prob
                if debug:
                    print("P({}|{}) = {}".format(current_word, prev_words, prob))
            else:
                # if the n-gram doesn't exist, return 0.0 or smooth the probability
                probability +=  np.log(1e-10)
                if debug:
                    print("P({}|{}) = {}".format(current_word, prev_words, 'N/A'))

        if(debug):
            print("Probability of the sentence: {}".format(probability))
        return probability
    
   
    def perplexity(self, sentence):
        probability = self.sentence_probability(sentence)
        K = len(self.preprocessSentence(sentence))
        perplexity = np.exp(-1/K * probability)
        return perplexity
   
    
    
# Create an instance of the n-gram language model with n=3 (trigram model)
model = NGramLanguageModelLogProbs(2)

# Train the model on a corpus (a list of sentences or words)
corpus =[
    'Peter Piper picked a peck of pickled pepper. ',
    "Where's the pickled pepper that Peter Piper picked?",
]
print(corpus)

model.train(corpus)
sentence = "Peter Piper picked a peck of pickled pepper."
sentence2 = "Peter Piper pepper asdasda."

print("Sentence Probability:", model.sentence_probability(sentence))
print("Sentence Probability:", model.sentence_probability(sentence2))

print("Input 1 Sentence Perplexity:", model.perplexity(sentence))
print("Input 2 Sentence Perplexity:", model.perplexity(sentence2))

# Generate new text using the model
seed = None
generated_text = model.generate(seed, max_length=12)
print('Generated Text:', (generated_text))

#model.printDataframe()