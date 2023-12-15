# Import the necessary libraries
import numpy as np
from io import open
from conllu import parse_incr

class MajorityTagger():
    
    def __init__(self,tags_set):
        self.emission_counts = {}
        self.tags = tags_set


    def fit(self, trainPath):
        file = open(trainPath, "r", encoding="utf-8")
        for sentence in parse_incr(file):
            # emission counts
            for token in sentence:
                word = token["form"];
                tag = token["lemma"];
                if(word not in self.emission_counts):
                    self.emission_counts[word]={}
                if(tag not in self.emission_counts[word]):
                    self.emission_counts[word][tag] = 0
                self.emission_counts[word][tag] += 1

                    


    def predict(self, tokens):
        predicted_tags = []
        for token in tokens:
            if token in self.emission_counts:
                predicted = self.emission_counts[token]
                most_common_tag = max(predicted, key=lambda k: predicted[k])

            else:
                most_common_tag = "O"
        
            predicted_tags.append((token, most_common_tag))

        return predicted_tags