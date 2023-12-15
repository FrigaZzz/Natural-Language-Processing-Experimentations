import numpy as np
from io import open
from conllu import parse_incr
import random

class RandomTagger():
    
    def __init__(self,tags_set,random_seed=123456):
        self.tags = tags_set
        random.seed(random_seed)


    def fit(self, dataPath):
        # doesn't learn anything
        pass


    def predict(self, tokens):
        predicted_tags = []
        for token in tokens:
            random_tag = random.choice(self.tags)

            predicted_tags.append((token, random_tag))

        return predicted_tags