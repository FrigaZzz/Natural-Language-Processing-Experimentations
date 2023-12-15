import nltk



# open the data.txt file and read the data
with open('data.txt','r', encoding='utf-8') as f:
    article = f.read()
    sentences = nltk.sent_tokenize(article)[1:]



#writing the sentences to a file
with open('sentences.txt', 'w') as file:
    for sentence in sentences:
        file.write(sentence + '\n')
        