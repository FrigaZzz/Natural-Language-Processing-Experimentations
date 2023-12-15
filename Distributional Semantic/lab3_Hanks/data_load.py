from nltk.corpus import semcor
from nltk import *
import nltk 
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from concurrent.futures import ThreadPoolExecutor
import pickle

lemmatizer = WordNetLemmatizer()

import json


def get_sentences_from_semcor(search_word, extract_tag_sense=False):
    '''
    :param search_word: the word that we are looking for
    :param extract_tag_sense: if true, the function will extract the tag sense of the word
    :return: a list of sentences that contain the search word
    '''
    sentences = []
    all_sentences = semcor.sents()

    # lemmatize the search word
    search_word_lemmatized = lemmatizer.lemmatize(search_word, 'v')  # 'v' stands for verb
    # join sentence tokens into a string
    selected_sentences = [(i, ' '.join(sentence)) for i, sentence in enumerate(all_sentences) 
                          if search_word_lemmatized in [lemmatizer.lemmatize(w.lower(), 'v') for w in sentence]][:4000]

    print(len(selected_sentences))

    # save as a json file all of the sentences with spaced words
    with open('sentences.json', 'w') as outfile:
        json.dump(selected_sentences, outfile, indent=4)

    if(extract_tag_sense):
        for i, sentenceElement in enumerate(selected_sentences):
            tagged_sentence = semcor.tagged_sents(tag='both')[sentenceElement[0]]
            tagged_sense = search_word_synset(tagged_sentence, search_word)
            if(tagged_sense is None):
                continue
            sentences.append(
                {
                    'sentence': sentenceElement[1],
                    'tagged_sense': tagged_sense
                }
            )
            print(len(sentences))
            if len(sentences) >= 200:
                break
        return sentences

def traverse_tree(tree, search_word):
    """Recursively traverse nltk.tree.Tree, find search_word and return its synset.

    Args:
        tree (nltk.tree.Tree): An instance of nltk.tree.Tree to traverse.
        search_word (str): The word to find in the tree.

    Returns:
        nltk.corpus.reader.wordnet.Synset: The synset of the search_word if found, None otherwise.
    """
    for idx, item in enumerate(tree):
        if isinstance(item, Tree):
            if 'V' not in item.label():
                continue  # Skip this subtree
            result = traverse_tree(item, search_word)
            if result is not None and result == search_word:
                # check if current node contans the label that is a Lemma class
                if item.label() is not None and isinstance(item.label(), nltk.corpus.reader.wordnet.Lemma):
                    return item.label().synset().name()

                if isinstance(tree[0], tuple):
                    return tree[0]
                else:
                    return result
        else:
            if isinstance(item, tuple):
                token, synset_id = item  # Unpack tuple
                if token == search_word:
                    if synset_id is not None:
                        synset = wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))
                        return synset
            else:
                token = item  # not a tuple, just a token
                if token == search_word:
                    return token
    return None

def search_word_synset(tagged_sentence,target_word):
    '''
    :param tagged_sentence: a tagged sentence from semcor
    :param target_word: the word that we are looking for
    :return: the synset of the target word
    '''
    for subtree in tagged_sentence:
        result = traverse_tree(subtree,target_word)
        if result is not None:
            if(result == target_word):
                if(isinstance(subtree.label(), nltk.corpus.reader.wordnet.Lemma)):
                    #check if label is in correct format and fix error if present
                    return subtree.label().synset().name()
                else: # the error that I have found is 'see.v.7;1'
                    return subtree.label().split(";")[0]
            return result


def test():
    search = "see"
    sentences = get_sentences_from_semcor(search)
    sentence = sentences[0]
    tagged_sentence =sentence['tagged_sentence']
    # print(tagged_sentence)
    # print(sentence['sentence'])
    print(sentences)


    # Target token
    target_token = "see"  


    # Find the semantic tag of the target token within the tagged sentence
    sem_tag = search_word_synset(tagged_sentence,target_token)

    if sem_tag is not None:
        print("Semantic tag of '{}': {}".format(target_token, sem_tag))
    else:
        print("The token '{}' is not present in the sentence.".format(target_token))

search = "see"
sentences = get_sentences_from_semcor(search)

'''
[['The'], 
Tree(Lemma('group.n.01.group'), [Tree('NE', ['Fulton', 'County', 'Grand', 'Jury'])]),
Tree(Lemma('state.v.01.say'), ['said']),
Tree(Lemma('friday.n.01.Friday'),['Friday']), ['an'],

Tree(Lemma('probe.n.01.investigation'), ['investigation']), ['of'], Tree(Lemma('atlanta.n.01.Atlanta'), ['Atlanta']), ["'s"], Tree(Lemma('late.s.03.recent'), ['recent']), Tree(Lemma('primary.n.01.primary_election'), ['primary', 'election']), Tree(Lemma('produce.v.04.produce'), ['produced']), ['``'], ['no'], Tree(Lemma('evidence.n.01.evidence'), ['evidence']), ["''"], ['that'], ['any'], Tree(Lemma('abnormality.n.04.irregularity'), ['irregularities']), Tree(Lemma('happen.v.01.take_place'), ['took', 'place']), ['.']] 

[('Fulton', Synset('group.n.01')), ('County', Synset('group.n.01')), ('Grand', Synset('group.n.01')), ('Jury', Synset('group.n.01')), ('said', Synset('state.v.01')), ('Friday', Synset('friday.n.01')), ('investigation', Synset('probe.n.01')), ('Atlanta', Synset('atlanta.n.01')), ('recent', Synset('late.s.03')), ('primary', Synset('primary.n.01')), ('election', Synset('primary.n.01')), ('produced', Synset('produce.v.04')), ('evidence', Synset('evidence.n.01')), ('irregularities', Synset('abnormality.n.04')), ('took', Synset('happen.v.01')), ('place', Synset('happen.v.01'))]



'''