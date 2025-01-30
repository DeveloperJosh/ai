import torch
import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json') as file:
    intents = json.load(file)

data = torch.load("chatbot_model.pth")
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = nltk.word_tokenize(msg)
    sentence = [lemmatizer.lemmatize(word.lower()) for word in sentence]
    X = [1 if word in sentence else 0 for word in all_words]
    X = np.array(X).reshape(1, -1)
    X = torch.from_numpy(X).float().to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "I didn't understand that."

if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        print("Bot:", get_response(sentence))