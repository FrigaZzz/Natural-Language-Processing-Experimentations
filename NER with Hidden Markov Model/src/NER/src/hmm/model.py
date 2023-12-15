# Import the necessary libraries
import numpy as np
from io import open
from conllu import parse_incr
import pyconll 
from collections import defaultdict


# Define the HMM model
class HiddenMarkovModel:
    ZERO_PROB = 0.00001
    
    # INIT DATA STRUCTURE
    def __init__(self, ner_tags):
        self.tagset = ner_tags
        self.totTags= len(ner_tags)
        self.transition=dict()
        self.transition_probs =  dict()
        self.emission_probs = dict()
        self.count_words= dict()
        self.statistics=np.zeros(len(ner_tags)) 
        self.count_tags = np.zeros(len(ner_tags) + 1, dtype = int);

    def fit(self, traing_path, tagName='lemma'):
        self.__init__(self.tagset) # Initialize the model

        # Count number of sentences for calculating probability of transition with the starting tag SQ0
        nSentences = 0;
    
        # Open the training file
        file = open(traing_path, "r", encoding="utf-8")

        # Iterate over each sentence in the file
        for sentence in parse_incr(file):
            prev_tag = 'Q0'; # Set the previous tag to Q0 at the start of each sentence
            nSentences = nSentences + 1; # Increment sentence counter for each new sentence

            # Iterate over each token in the sentence
            for token in sentence:
                word = token["form"]; # Get the word for the token
                tag = token[tagName]; # Get the tag for the token

                # Count the word/tag pair and calculate emission probability
                self.count(word, tag);
                self.calculateEmissionProbability(word,tag);

                # Calculate transition probability using the previous tag and the current tag
                self.calculateTransitionProbability(prev_tag, tag, nSentences)

                # Set the previous tag to the current tag for the next iteration
                prev_tag = tag           

        # Return the calculated probabilities and statistics
        return self.emission_probs, self.transition_probs, self.statistics;

    def predict(self,words,smoothingStrategy=0):
        # Initialize variables
        start_tag = "Q0";
        viterbi_matrix = np.zeros((self.totTags,len(words)));
        backtrace = np.zeros(len(words), dtype = int);
        probabilites = np.zeros(len(words));
    
        # First iteration of Viterbi algorithm to initialize first column
        for tag_idx, tag in enumerate(self.tagset):
            # Get emission probability for the first word (here we can apply smoothing)
            probE = self.selectSmoothing(words[0],tag_idx,smoothingStrategy)
            if probE == 0:
                # if probE is zero, set it to a small value to avoid log(0) error
                probE = np.log(0.00001)
            else:
                probE = np.log(probE)
            # Set initial transition probability for  Q0
            probT = np.log(0.00001);
            tran_tag = "%s_%s" % (tag,start_tag);   
            # Calculate initial viterbi probability for each tag
            if tran_tag in self.transition_probs.keys():
                probT = np.log(self.transition_probs[tran_tag]);
            viterbi_matrix[tag_idx][0] = probE + probT;    
            
        index_max_values = np.argmax(viterbi_matrix[:,0]);  
        backtrace[0] = index_max_values;
        probabilites[0] = viterbi_matrix[index_max_values,0];
        
        # Update Viterbi matrix for each word in sentence
        t= 1;
        for word in words[1:]:
            # Calculate viterbi probability for each tag
            for tag_idx, tag in enumerate(self.tagset):
                # Get emission probability for current word (here we can apply smoothing)
                probE = self.selectSmoothing(word, tag_idx, smoothingStrategy);
                if probE == 0:
                    probE = np.log(HiddenMarkovModel.ZERO_PROB)
                else:
                    probE = np.log(probE)
                
                # Evaluate "max vt-1*aij*bj" between the N tags to get the maximum 
                # viterbi probability for current tag
                max_tmp = np.zeros(self.totTags);
                for i in range(0,self.totTags):
                    tran_tag = "%s_%s" % (tag,self.tagset[i]);
                    probT = np.log(HiddenMarkovModel.ZERO_PROB);
                    if tran_tag in self.transition_probs.keys():
                        probT = np.log(self.transition_probs[tran_tag]);
                    max_tmp[i] = viterbi_matrix[i,t-1] + probT;
                # Update Viterbi matrix with the maximum probability for current tag
                viterbi_matrix[tag_idx,t] = np.max(max_tmp) + probE;
                
            # Update backtrace and probability lists with the max computed
            index_max_values = np.argmax(viterbi_matrix[:,t]);  
            backtrace[t] = index_max_values;
            probabilites[t] = viterbi_matrix[index_max_values,t];
            t= t +1;
        return backtrace,probabilites;
     
    def count(self, word, tag):
        index_of_tag = self.tagset.index(tag)
        
        # increment count of this tag
        self.count_tags[index_of_tag] = self.count_tags[index_of_tag] + 1
        
        # increment count of this word for this tag
        if word in self.count_words.keys():
            self.count_words[word][index_of_tag] = self.count_words[word][index_of_tag] + 1
        else:
            # create a new row for the word
            word_row = np.zeros(self.totTags + 1, dtype=int)
            word_row[index_of_tag] = 1
            self.count_words[word] = word_row
        
        # increment total count of this word
        self.count_words[word][self.totTags] = self.count_words[word][self.totTags] + 1
        
        # increment total count of tags
        self.count_tags[self.totTags] = self.count_tags[self.totTags] + 1
    
    def calculateEmissionProbability(self, word, tag):
        # Get index of tag in tagset
        index_of_tag = self.tagset.index(tag)

        # Calculate emission probability of word given tag
        emissionProb = self.count_words[word][index_of_tag] / self.count_tags[index_of_tag]

        # If word already exists in emission_probs, update its probability for the current tag
        if(word in self.emission_probs.keys()):
            self.emission_probs[word][index_of_tag] = emissionProb

        # If word doesn't exist in emission_probs, create a new row for the word and initialize with the current tag's  probability
        else:
            prob_row = np.zeros(self.totTags + 1)
            prob_row[index_of_tag] = emissionProb
            self.emission_probs[word] = prob_row;

    def calculateTransitionProbability(self, prev_tag, tag,  nSentences):
        #natural sequence when scanning senteces
        prev_tag_to_tag = "%s_%s" % (prev_tag,tag);
        #Sequence used to saved
        tag_to_prev_tag = "%s_%s" % (tag,prev_tag);
        
        if(prev_tag_to_tag in self.transition.keys()):
            self.transition[prev_tag_to_tag] = self.transition[prev_tag_to_tag] + 1;
        else:
            self.transition[prev_tag_to_tag] = 1;

        if(prev_tag_to_tag in self.transition_probs.keys() and prev_tag != 'Q0'):
            index_of_tag = self.tagset.index(prev_tag);
            self.transition_probs[tag_to_prev_tag] = self.transition[prev_tag_to_tag] /self.count_tags[index_of_tag];
        else:
            self.transition_probs[tag_to_prev_tag] = self.transition[prev_tag_to_tag] / nSentences;    

    def selectSmoothing(self,word, index_of_tag,smoothingStrategy):
        # NO SMOOTHING: if the word exists in the emission probabilities, return its corresponding probability
        if word in self.emission_probs.keys():
           return self.emission_probs[word][index_of_tag]
        # STRATEGY 0: if the word doesn't exist in the emission probabilities, return a zero  probability
        elif smoothingStrategy == 0 and word not in self.emission_probs.keys():
            return HiddenMarkovModel.ZERO_PROB;
         # STRATEGY 1: if the word doesn't exist in the emission probabilities, set the probability      of tag "O" to 1
        elif smoothingStrategy == 1 and word not in self.emission_probs.keys():
            prob_row = np.zeros(self.totTags + 1);
            index_of_o = self.tagset.index("O");
            prob_row[index_of_o] = 1;
            return prob_row[index_of_tag];
        # STRATEGY 2: if the word doesn't exist in the emission probabilities, set the probabilities of tags "O", "B-MISC", and "I-MISC" to 0.33
        elif smoothingStrategy == 2 and word not in self.emission_probs.keys():
            prob_row = np.zeros(self.totTags + 1);
            index_of_i_misc = self.tagset.index("I-MISC");
            index_of_b_misc = self.tagset.index("B-MISC");
            index_of_o = self.tagset.index("O");
            prob_row[index_of_i_misc] = 0.33;
            prob_row[index_of_b_misc] = 0.33;
            prob_row[index_of_o] = 0.33;
            return prob_row[index_of_tag];
        # STRATEGY 3: if the word doesn't exist in the emission probabilities, set the probability      of each tag to 1/total number of tags
        elif smoothingStrategy == 3 and word not in self.emission_probs.keys():
            unk_prob = 1 / self.totTags;
            prob_row = np.full(self.totTags + 1, unk_prob);
            return prob_row[index_of_tag];
        # STRATEGY 4: if the word doesn't exist in the emission probabilities, use thestatistics of the POS tagger for every word in the development set that has oneoccurrence for each tag
        elif smoothingStrategy == 4 and word not in self.emission_probs.keys():
            return self.statistics[index_of_tag];
        # if none of the above conditions apply, return a zero probability
        else:
            return HiddenMarkovModel.ZERO_PROB;

    def calculateStatisticPosTagging(self,evalPath):
        # create a numpy array to keep track of the count of tags that occur only once
        unique_tags_occourances = np.zeros(self.totTags, dtype = int);
        # initialize a counter for the total number of tags that occur only once
        unique_tags = 0;
    
        # open the evaluation file and parse it
        file = open(evalPath, "r", encoding="utf-8")
        for sentence in parse_incr(file):
            # iterate through each token in the sentence and count the word/tag occurrences
            for token in sentence:
                word = token["form"];
                tag = token["lemma"];
                self.count(word, tag);
    
        # iterate through each word in the count_words dictionary
        for word in self.count_words:
            # if the word occurs only once, update the count of its tag
            if self.count_words[word][self.totTags] == 1:
                # find the tag with the highest count for the word
                index_of_tag = np.argmax(self.count_words[word]);
                # increment the count of the tag that occurs only once
                unique_tags_occourances[index_of_tag] = unique_tags_occourances[index_of_tag] + 1;
                # increment the total count of tags that occur only once
                unique_tags =  unique_tags + 1;
    
        # calculate the statistics for each tag in the tagset
        for tag in self.tagset:
            # get the index of the current tag in the tagset
            index_of_tag = self.tagset.index(tag);
            # calculate the probability of a tag occurring only once for the current tag
            self.statistics[index_of_tag] = unique_tags_occourances[index_of_tag] / unique_tags;


