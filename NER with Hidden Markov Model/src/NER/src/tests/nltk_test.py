from nltk.tag import hmm
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import os
from conllu import parse

def getDataPath(language,type):
    return ""+os.path.join(os.path.abspath('.'), "assets\\wikineural_corpus\%s\%s.conllu" % (language,type))
def readFile(language,fileType):
    nameFile = getDataPath(language,fileType)
    tsv_file = open(nameFile,"r",encoding="utf-8").read();
    sentences = parse(tsv_file)
    return sentences;
def tokenize_sentence(sentence):
    words = [];
    for token in sentence:
        words.append(token["form"]);
    return words

# Load and preprocess the data / { it , en } 
train_data = readFile("en","train")
test_data = readFile("en","test")
train_data2=[]
test_data2=[]
for sentence in train_data:
    words = tokenize_sentence(sentence)
    tags = [token["lemma"] for token in sentence]
    train_data2.append(list(zip(words, tags)))
for sentence in test_data:
    words = tokenize_sentence(sentence)
    tags = [token["lemma"] for token in sentence]
    test_data2.append(list(zip(words, tags)))  
    
train_data2=list(train_data2)
test_data2=list(test_data2)
tags_italian = ['B-PER','B-ORG','B-LOC','B-MISC','I-PER','I-ORG','I-LOC','I-MISC','O']

trainer = hmm.HiddenMarkovModelTrainer(symbols=tags_italian)
tagger = trainer.train_supervised(train_data2)
from sklearn.metrics import accuracy_score

y_true = []
y_pred = []
for sentence in test_data2:
    tokens, true_tags = zip(*sentence)
    predicted_tags = [tag for _, tag in tagger.tag(tokens)]

    y_true.extend(true_tags)
    y_pred.extend(predicted_tags)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"POS tagger accuracy: {accuracy:.2%}")
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)
# print the results
print("Accuracy: {:.3f}".format(accuracy))
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("F1-score: {:.3f}".format(f1_score))