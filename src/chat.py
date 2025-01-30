import torch
import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('data/intents.json') as file:
    intents = json.load(file)

# Load model data with security setting
data = torch.load("chatbot_model.pth", map_location=device, weights_only=True)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Updated model architecture to match training
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def clean_input(pattern):
    pattern = re.sub(r'[^\w\s]', '', pattern)
    tokens = nltk.word_tokenize(pattern)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def get_response(msg):
    sentence = clean_input(msg)
    X = [1 if word in sentence else 0 for word in all_words]
    X = np.array(X, dtype=np.float32).reshape(1, -1)
    X = torch.from_numpy(X).to(device)
    
    with torch.no_grad():
        output = model(X)
    
    _, predicted = torch.max(output, 1)
    tag = tags[predicted.item()]
    
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "I didn't understand that. Could you rephrase that?"

if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        print("Bot:", get_response(sentence))