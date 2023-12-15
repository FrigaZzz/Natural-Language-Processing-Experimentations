from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from src.hmm.memm_tagger import  * 
import os

path =os.path.abspath('.')

train_d = ""+os.path.join(path,".\\assets\\wikineural_corpus\\en\\train.txt")
test_d =""+os.path.join(path,".\\assets\\wikineural_corpus\\en\\test.txt")

data = initialize()
log_reg = train(train_d, data)
print_message("Test Model")
test(test_d, log_reg, data)

