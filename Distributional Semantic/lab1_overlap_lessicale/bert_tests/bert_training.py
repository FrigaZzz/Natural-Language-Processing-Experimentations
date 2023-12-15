import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from torch.optim import AdamW
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup

# 1. Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 2. Load and preprocess your dataset using the datasets library
file_path = "./preprocessed_sentences.txt"  # replace with the path to your file

dataset = load_dataset("text", data_files=file_path)["train"]

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_data = dataset.map(tokenize_function, batched=True)

# 3. Create a data collator that will be used for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 4. Initialize the model with transfer learning
model = BertForMaskedLM(BertConfig())
device = torch.device('cuda') 
model.to(device)

# 5. Initialize the Trainer
training_args = TrainingArguments(
    output_dir="./out",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=10,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,  # Adjust the learning rate
    weight_decay=0.0001,  # Apply weight decay as regularization
)

# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(tokenized_data) * training_args.num_train_epochs
)

# 6. Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_data,
    optimizers=(optimizer, scheduler),
)

trainer.train()


# 7. Save the model
trainer.save_model("./bert_from_scratch")
