import numpy as np
import torch
import torch.nn as nn
import json
import random
from torch.utils.data import Dataset, DataLoader
import nltk
import re

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Preprocessing
with open('data/intents.json') as file:
    intents = json.load(file)

lemmatizer = nltk.stem.WordNetLemmatizer()
ignore_chars = ['?', '!', '.', ',', '*']

def clean_tokenize(pattern):
    # Remove special characters first
    pattern = re.sub(r'[^\w\s]', '', pattern)
    tokens = nltk.word_tokenize(pattern)
    # Lemmatize and clean
    return [lemmatizer.lemmatize(word.lower()) 
            for word in tokens if word not in ignore_chars]

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Clean and tokenize each pattern
        tokens = clean_tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []

for (pattern_tokens, tag) in xy:
    bag = [1 if word in pattern_tokens else 0 for word in all_words]
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

# Hyperparameters (adjusted for better convergence)
batch_size = 16
input_size = len(X_train[0])
hidden_size = 256
output_size = len(tags)
learning_rate = 0.0005
num_epochs = 1500
dropout_prob = 0.3

# Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Enhanced Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

# Training with early stopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

best_loss = float('inf')
patience = 50
no_improve = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    
    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
    else:
        no_improve += 1
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
    if no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

print(f'Best loss: {best_loss:.4f}')

# Save model and data
torch.save({
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}, "chatbot_model.pth")

print(f'Training complete. Model saved to chatbot_model.pth')