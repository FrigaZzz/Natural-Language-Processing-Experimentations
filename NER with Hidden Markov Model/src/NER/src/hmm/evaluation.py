import numpy as np
import os
from conllu import parse
from .model import HiddenMarkovModel 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def getDataPath(language,type):
    return ""+os.path.join(os.path.abspath('.'), "NER\\assets\wikineural_corpus\%s\%s.conllu" % (language, type))
def readFile(language,fileType):
    nameFile = getDataPath(language,fileType)
    tsv_file = open(nameFile,"r",encoding="utf-8").read();
    sentences = parse(tsv_file)
    return sentences;

def readFileFromName(path):
    tsv_file = open(path,"r",encoding="utf-8").read();
    sentences = parse(tsv_file)
    return sentences;

def tokenize_sentence(sentence):
    return [token["form"] for token in sentence]

def evaluate(model: HiddenMarkovModel, language, tagset, smoothing_type):
    sentences_test = readFile(language, "test")
    count_true_positive = np.zeros(len(tagset), dtype=int)
    count_predicted = np.zeros(len(tagset), dtype=int)
    count_actual = np.zeros(len(tagset), dtype=int)
    y_true = []
    y_pred = []
    errors = dict();

    for sentence in sentences_test:
        words = tokenize_sentence(sentence)
        backtrace_test, probabilities_test = model.predict(words, smoothing_type)
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

    return confusion_mat,accuracy, precision, recall, f1_score

def evaluateBaseLines(model: any, language, tagset):
    sentences_test = readFile(language, "test")
    count_true_positive = np.zeros(len(tagset), dtype=int)
    count_predicted = np.zeros(len(tagset), dtype=int)
    count_actual = np.zeros(len(tagset), dtype=int)
    y_true = []
    y_pred = []
    errors = dict();

    for sentence in sentences_test:
        words = tokenize_sentence(sentence)
        predicted = model.predict(words)
        y_true.extend( [token["lemma"]] for token in sentence)
        y_pred.extend( [eval[1]] for eval in predicted)
                
    # using sklearn metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    # print the results
    print("Accuracy: {:.3f}".format(accuracy))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1-score: {:.3f}".format(f1_score))

    return errors,accuracy, precision, recall, f1_score
