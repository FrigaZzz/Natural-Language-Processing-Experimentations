import os
# Script used to convert the CONLLU file in the format that 
# is used by memm_tagger algo

# define input and output file paths
input_file = "C:\\Users\\User\\Documents\\repos\\TLN\\1.MAZZEI\\NER\\src\\hmm\\NER\\assets\\wikineural_corpus\\it\\val.conllu"
output_file = "C:\\Users\\User\\Documents\\repos\\TLN\\1.MAZZEI\\NER\\src\\hmm\\NER\\assets\\wikineural_corpus\\it\\val.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_file, "w", encoding="utf-8") as f:
    for line in lines:
        # Strip the first element and keep the other two separated by a tab
        elements = line.strip().split("\t")[1:]
        f.write("\t".join(elements) + "\n")
