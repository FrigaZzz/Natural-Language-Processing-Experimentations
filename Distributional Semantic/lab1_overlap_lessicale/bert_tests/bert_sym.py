from transformers import BertForMaskedLM, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

model_path = "./bert_from_scratch"  # Path to the saved BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(model_path)
model.eval()
device = torch.device('cpu')
model.to(device)

# Example sentences
sentence1 = "It's an opening, it can be opened or closed."
sentence2 = "A construction used to divide two rooms, temporarily closing the passage between them"
sentence3 = "suffer suffer suffer suffer suffer suffer suffer"

# Tokenize and convert sentences to token IDs
tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)
tokens3 = tokenizer.tokenize(sentence3)

input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
input_ids3 = tokenizer.convert_tokens_to_ids(tokens3)

# Convert token IDs to tensors
input_ids1 = torch.tensor(input_ids1).unsqueeze(0).to(device)
input_ids2 = torch.tensor(input_ids2).unsqueeze(0).to(device)
input_ids3 = torch.tensor(input_ids3).unsqueeze(0).to(device)

# Obtain the embeddings for each token
with torch.no_grad():
    outputs1 = model(input_ids=input_ids1)
    token_embeddings1 = outputs1[0].squeeze(0)
    outputs2 = model(input_ids=input_ids2)
    token_embeddings2 = outputs2[0].squeeze(0)
    outputs3 = model(input_ids=input_ids3)
    token_embeddings3 = outputs3[0].squeeze(0)

# Calculate the mean pooling of token embeddings to obtain the sentence embeddings
sentence_embedding1 = torch.mean(token_embeddings1, dim=0)
sentence_embedding2 = torch.mean(token_embeddings2, dim=0)
sentence_embedding3 = torch.mean(token_embeddings3, dim=0)

# Normalize the embeddings
normalized_embedding1 = sentence_embedding1 / sentence_embedding1.norm()
normalized_embedding2 = sentence_embedding2 / sentence_embedding2.norm()
normalized_embedding3 = sentence_embedding3 / sentence_embedding3.norm()

# Calculate the cosine similarity
similarity12 = cosine_similarity(normalized_embedding1.unsqueeze(0), normalized_embedding2.unsqueeze(0))
similarity13 = cosine_similarity(normalized_embedding1.unsqueeze(0), normalized_embedding3.unsqueeze(0))
similarity23 = cosine_similarity(normalized_embedding2.unsqueeze(0), normalized_embedding3.unsqueeze(0))

# Check if the sentences are close based on cosine similarity
print(similarity12)
print(similarity13)
print(similarity23)
