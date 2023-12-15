# Import necessary libraries and modules
from collections import defaultdict
import random
import math
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

class NGramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "probability": 0.0}))
    
    def preprocessSentence(self, sentence):
        # Tokenize the sentence
        cleaned_words = word_tokenize(sentence)

        # Remove punctuation from each word
        cleaned_words = [re.sub(r"[^\w\s]", "", word)   for word in cleaned_words if re.sub(r"[^\w\s]", "", word) is not '']

        return cleaned_words


    def train(self, corpus):
        # Preprocess corpus to add start and end tokens
        # + PADDING of the sentence -> <s> and </s> are added two markers that delimit the sentences
        processed_corpus = []
        for sentence in corpus:
            
            tokenized_sentence = self.preprocessSentence(sentence)
            # add padding
            processed_sentence = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']
            processed_corpus.extend(processed_sentence)

        # Generate n-grams from the corpus
        for i in range(len(processed_corpus) - self.n + 1):
            prev_words = tuple(processed_corpus[i:i+self.n-1])
            current_word = processed_corpus[i+self.n-1]

            self.ngrams[prev_words][current_word]['frequency'] += 1
            self.ngrams[prev_words][current_word]['probability'] += 1
    

        # Normalize counts to estimate probabilities using MLE from self.ngrams
            
        for prev_words, next_words in self.ngrams.items():
            total_count = sum(next_words[word]['frequency'] for word in next_words)
            for word in next_words:
                self.ngrams[prev_words][word]['probability'] = next_words[word]['frequency'] / total_count


    def printDataframe(self):
        # Flatten the nested dictionaries into separate columns
        data = [(prev_words, len(next_words), next_words) for prev_words, next_words in self.ngrams.items()]
        df = pd.DataFrame(data, columns=["prev_words", "frequency", "next_words"])
        df = pd.concat([df.drop(['next_words'], axis=1), json_normalize(df['next_words'])], axis=1)

        # Set the maximum display width for columns
        pd.set_option("display.max_colwidth", None)

        # Display the DataFrame as an Excel-like table
        display(df)
        df.to_csv('ngram.csv', index=False)
       
    def sentence_probability(self, sentence,debug=False):
        # Preprocess the input sentence
        tokenized_sentence = self.preprocessSentence(sentence)
        processed_sentence = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']

        # Initialize probability to maximum in logaritmic space
        probability = 1.0
        
        
        # Iterate over the sentence to compute the probability
        for i in range(len(processed_sentence) - self.n + 1):
            prev_words = tuple(processed_sentence[i:i+self.n-1])
            current_word = processed_sentence[i+self.n-1]

            # Check if the n-gram exists in the language model
            if prev_words in self.ngrams and current_word in self.ngrams[prev_words]:
                # Multiply the probability by the conditional probability of the current word given the previous words
                prob = self.ngrams[prev_words][current_word]['probability']
                log_prob = np.log(prob)
                probability *= prob
                if debug:
                    print("P({}|{}) = {}".format(current_word, prev_words, prob))
            else:
                # if the n-gram doesn't exist, return 0.0 or smooth the probability
                probability *=0.0000000001
                if debug:
                    print("P({}|{}) = {}".format(current_word, prev_words, 'N/A'))

        if(debug):
            print("Probability of the sentence: {}".format(probability))
        return probability
    
    def getFrequency(self):
        return self.ngrams
    
    def generate(self, seed=None, max_length=10, top_k=5):
        detect_loop=15 
        if seed is None:
            seed = ['<s>'] * (self.n - 1)
        
        prev_words = tuple(seed)[-(self.n - 1):]
        sentence = list(seed)
        
        while len(sentence) < max_length:
            if(detect_loop==0):
                break
            possible_next_words = self.ngrams[tuple(prev_words)]
            if not possible_next_words:
                break
            
            # Select the top-k most probable words
            top_words = sorted(possible_next_words.keys(),
                               key=lambda word: possible_next_words[word]['probability'],
                               reverse=True)[:top_k]
    
            next_word = random.choice(list(top_words))
            
            if next_word in sentence:
                # If the next word is already in the sentence, skip it and continue
                detect_loop-=1
                continue
            else:
                detect_loop=15

            # Append the selected word to the sentence
            sentence.append(next_word)
            
            # Update the previous words for the next iteration
            # Remove the first word and add the selected word at the end
            prev_words = prev_words[1:] + (next_word,)
            
        
        return sentence



# Create an instance of the n-gram language model with n=3 (trigram model)
model = NGramLanguageModel(2)

# Train the model on a corpus (a list of sentences or words)
# Train the model on a corpus (a list of sentences or words)
corpus =[
    'Peter Piper picked a peck of pickled pepper. ',
    "Where's the pickled pepper that Peter Piper picked?",
]
print(corpus)

model.train(corpus)

sentence = "Peter Piper picked"

probability = model.sentence_probability(sentence,debug=True)
print("Sentence Probability:", probability)

# Generate new text using the model
seed = None
generated_text = model.generate(max_length=20)
print('Generated Text:', (generated_text))
#model.printDataframe()
