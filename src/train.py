import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
with open('data/intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
ignore_chars = ['?', '!', '.', ',']

def clean_tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in ignore_chars]

# Prepare data
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = clean_tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create word-to-index mappings
word2idx = {word: idx for idx, word in enumerate(all_words)}
idx2word = {idx: word for idx, word in enumerate(all_words)}
tag2idx = {tag: idx for idx, tag in enumerate(tags)}
idx2tag = {idx: tag for idx, tag in enumerate(tags)}

# Convert data to tensors
X = []
y = []

for (pattern_tokens, tag) in xy:
    X.append([word2idx[word] for word in pattern_tokens])
    y.append(tag2idx[tag])

# Pad sequences
max_len = max(len(seq) for seq in X)
X = [seq + [0] * (max_len - len(seq)) for seq in X]
X = torch.tensor(X, dtype=torch.long).to(device)  # Move to device
y = torch.tensor(y, dtype=torch.long).to(device)  # Move to device

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        embedded = self.encoder(x)
        output, hidden = self.rnn(embedded)
        decoded = self.decoder(output)
        return decoded, hidden

# Hyperparameters
input_size = len(all_words)
hidden_size = 128
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Initialize model, loss, and optimizer
model = Seq2Seq(input_size, hidden_size, output_size).to(device)  # Fixed device assignment
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs, _ = model(X)
    loss = criterion(outputs.view(-1, output_size), y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model
torch.save({
    "model_state": model.state_dict(),
    "word2idx": word2idx,
    "idx2word": idx2word,
    "tag2idx": tag2idx,
    "idx2tag": idx2tag,
    "all_words": all_words,
    "tags": tags
}, "seq2seq_chatbot.pth")

print("Training complete. Model saved to seq2seq_chatbot.pth")