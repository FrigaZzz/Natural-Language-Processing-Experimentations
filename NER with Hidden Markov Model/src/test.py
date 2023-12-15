import numpy as np
from conllu import parse
from NER.src.hmm.model import HiddenMarkovModel
import numpy as np
import os
from conllu import parse
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from NER.src.hmm.evaluation import *
tagset = ['B-PER','B-ORG','B-LOC','B-MISC','I-PER','I-ORG','I-LOC','I-MISC','O']

# Italian Datasets
it_trainDataPath = getDataPath("it","train")
it_validateDataPath = getDataPath("it","val")
it_testDataPath = getDataPath("it","test")

# Training of the model

model= HiddenMarkovModel(tagset)

model.fit(it_trainDataPath)
model.calculateStatisticPosTagging(it_validateDataPath)

smoothing_type = 0
sentences_test = readFile("it", "test")
count_true_positive = np.zeros(len(tagset), dtype=int)
count_predicted = np.zeros(len(tagset), dtype=int)
count_actual = np.zeros(len(tagset), dtype=int)
y_true = []
y_pred = []
errors = dict();
for sentence in sentences_test:
    words = tokenize_sentence(sentence)
    backtrace_test, probabilities_test = model.predict2(words, smoothing_type)
    y_true.extend( [token["lemma"]] for token in sentence)
    y_pred.extend( [tagset[eval]] for eval in backtrace_test)
    for i, token in enumerate(sentence):
        real_tag = token["lemma"]
        real_tag_index = tagset.index(real_tag)
        predicted_tag_index = backtrace_test[i]
        count_actual[real_tag_index] += 1
        count_predicted[predicted_tag_index] += 1
        if real_tag_index == predicted_tag_index:
            count_true_positive[real_tag_index] += 1
        else:
            errors[token["form"]] = [real_tag,tagset[backtrace_test[i]]];
            
# using sklearn metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)
confusion_mat = confusion_matrix(y_true,y_pred, labels=tagset)
# print the results
print("Accuracy: {:.3f}".format(accuracy))
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("F1-score: {:.3f}".format(f1_score))