# Natural Language Processing Experimentations

Welcome to the Natural Language Processing Experimentations repository! This repository contains a collection of experiments and projects related to natural language processing (NLP). The goal is to explore various NLP techniques, algorithms, and applications to enhance understanding and proficiency in the field.


## Experiments

### Word Sense Disambiguation with Wordnet

Explore Word Sense Disambiguation using WordNet. Implement strategies to disambiguate word senses based on context.

### N-Gram Language Model for Trump Tweets

Develop an N-Gram Language Model to analyze language patterns in Trump's tweets. Explore the use of N-Grams for language understanding.

### Named Entity Recognition with Hidden Markov Model

Implement Named Entity Recognition using a Hidden Markov Model. Identify and classify entities in text based on their hidden states.

### Distributional Semantics Labs

#### Lab-1: Lexical Overlap Measurement

This lab involves calculating 2-to-2 similarity between a series of definitions for generic/specific and concrete/abstract concepts. Utilize the provided data in the "dati" folder on Moodle to compute pairwise similarity, for instance, using the cardinality of the intersection of normalized lemmas over the minimum length of definitions. Aggregate and average the similarity scores across both dimensions (concreteness/specificity).

#### Lab-2: Onomasiological Search System

This exercise explores building a system based on onomasiological search principles. Using definition data, construct a system that automatically identifies the target term by leveraging the multiplicity of definitions. Attempt various solutions, including definition filtering mechanisms (e.g., excluding less informative ones), searching in the WordNet taxonomic tree (using the Genus-Differentia principle), etc.

#### Lab-3: Valency Implementation

Implement Patrick Hanks' valency theory. Choose a corpus and a specific verb, then build possible semantic clusters with frequencies. For example, given the verb "to see" with valency = 2, use a syntactic parser (e.g., Spacy) to collect fillers for the subj and obj roles, converting them into semantic types. Generate frequent clusters such as subj = noun.person and obj = noun.artifact based on the chosen verb and corpus.

#### Lab-4a (Alternative to Lab-4b): Text Segmentation System

Implement a text segmentation system inspired by TextTiling. Use a corpus with at least 3 sections on vastly different topics. Test the system to identify appropriate cutting points.


#### Lab-5: Topic Modeling

Implement a Topic Modeling exercise using open libraries like Gensim. Use a corpus of at least 1k documents and test algorithms (e.g., LDA) with different values of k (number of topics). Evaluate results for coherence through fine-tuning parameters and pre-processing. Consider filtering only nouns during pre-processing for improved interpretability.

#### Lab-6: Module Development

Develop a ranking system based on basicness scores for WordNet synsets.


## Usage

Each experiment is contained in its own directory with a dedicated README providing specific instructions. Navigate to the experiment directory of interest and follow the instructions to run the experiment inside the specific Jupyter notebook.

