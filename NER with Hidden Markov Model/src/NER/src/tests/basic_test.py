import numpy as np
import os
from conllu import parse
import math
import nltk
from src.hmm.model import HiddenMarkovModel 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from NER.src.hmm.evaluation import *


tags = ['B-PER','B-ORG','B-LOC','B-MISC','I-PER','I-ORG','I-LOC','I-MISC','O']
hmm= HiddenMarkovModel(tags)

     
dataPath = getDataPath("en","train")
evalPath = getDataPath("en","val")

hmm.fit(dataPath);
hmm.calculateStatisticPosTagging(evalPath)
errors1, accuracy, avg_precision, avg_recall,    avg_f1_score = evaluate(hmm,"en",tags,0)
