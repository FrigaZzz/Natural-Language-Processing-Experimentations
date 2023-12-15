import csv
'''
    This utility is used to preprocess the TSV file containing the definitions of the words:
    - It extracts the sentences from the TSV file
    - It writes the sentences to a new file, one sentence per line
    The sentences are then used as input to the BERT model. (since the load_dataset function expects one sentence per line)
'''
# Path to the TSV file
tsv_file = "./../TLN-definitions-23.tsv"

# Output file to save preprocessed sentences
output_file = "preprocessed_sentences.txt"

# Open the TSV file and create the output file
with open(tsv_file, 'r', encoding='utf-8') as tsv, open(output_file, 'w', encoding='utf-8') as output:
    reader = csv.reader(tsv, delimiter='\t')
    # Skip the header row (words names)
    next(reader, None)

    # Iterate over each row in the TSV file
    for row in reader:
        # Extract the sentences from the row
        sentences = row[1:]

        # Write each sentence to the output file on a separate line
        for sentence in sentences:
            output.write(sentence.strip() + "\n")
