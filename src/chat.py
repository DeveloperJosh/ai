import torch
import random
import json
import numpy as np
import torch.nn as nn
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and vocabulary
data = torch.load("seq2seq_chatbot.pth", map_location=device)
word2idx = data["word2idx"]
idx2word = data["idx2word"]
tag2idx = data["tag2idx"]
idx2tag = data["idx2tag"]
all_words = data["all_words"]
tags = data["tags"]

# Load intents
with open('data/intents.json') as file:
    intents = json.load(file)

# Characters to ignore during tokenization
ignore_chars = set(['?', '!', '.', ','])

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

# Initialize and load the model
model = Seq2Seq(len(all_words), 128, len(tags)).to(device)
model.load_state_dict(data["model_state"])
model.eval()

# Tokenize and clean input text
def clean_tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in ignore_chars]

# Get chatbot response
def get_response(msg):
    tokens = clean_tokenize(msg)
    X = [word2idx[word] for word in tokens if word in word2idx]
    
    # Handle empty input sequence
    if len(X) == 0:
        return "I didn't understand that. Can you rephrase?"
    
    # Convert to tensor and move to device
    X = torch.tensor([X], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs, _ = model(X)
        _, predicted = torch.max(outputs, 2)
        tag = idx2tag[predicted[0][-1].item()]
    
    # Find the corresponding intent and return a random response
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "I didn't understand that. Can you rephrase?"

# Main loop to run the chatbot
if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        print("Bot:", get_response(sentence))