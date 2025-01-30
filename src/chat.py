import torch
import torch.nn as nn
import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Define special tokens
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and vocabulary
try:
    data = torch.load("seq2seq_chatbot.pth", map_location=device)
    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    all_words = data["all_words"]
except FileNotFoundError:
    raise FileNotFoundError("The model file 'seq2seq_chatbot.pth' was not found.")
except KeyError as e:
    raise KeyError(f"Missing key in the model data: {e}")

# Ensure special tokens exist in the vocabulary
for token in [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
    if token not in word2idx:
        word2idx[token] = len(word2idx)
        idx2word[len(idx2word)] = token
        all_words.append(token)

# Define the Seq2Seq Model for Response Generation
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.encoder_rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        # Decoder
        self.decoder = nn.Embedding(output_size, hidden_size)
        self.decoder_rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # Encode the source sequence
        embedded_src = self.encoder(src)
        encoder_outputs, hidden = self.encoder_rnn(embedded_src)

        # Prepare the target sequence
        embedded_trg = self.decoder(trg)

        # Decode the target sequence
        outputs, hidden = self.decoder_rnn(embedded_trg, hidden)
        predictions = self.fc_out(outputs)
        predictions = self.softmax(predictions)

        return predictions

# Initialize and load the model
input_size = len(all_words)
hidden_size = 256  # You can adjust this based on your model's architecture
output_size = len(all_words)
num_layers = 2  # Typically 2 layers work well for Seq2Seq models

model = Seq2Seq(input_size, hidden_size, output_size, num_layers).to(device)

try:
    model.load_state_dict(data["model_state"])
except KeyError:
    raise KeyError("The model state 'model_state' was not found in the loaded data.")
except RuntimeError as e:
    raise RuntimeError(f"Error loading the model state: {e}")

model.eval()

# Tokenize and clean input text
def clean_tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in set(['?', '!', '.', ','])]

# Generate response using Greedy Decoding
def generate_response(model, sentence, word2idx, idx2word, max_length=20):
    model.eval()
    with torch.no_grad():
        # Tokenize and preprocess the input sentence
        tokens = clean_tokenize(sentence)
        X = [word2idx.get(word, word2idx[UNK_TOKEN]) for word in tokens]
        X = torch.tensor([X], dtype=torch.long).to(device)

        # Encode the input sequence
        embedded_src = model.encoder(X)
        encoder_outputs, hidden = model.encoder_rnn(embedded_src)

        # Initialize the decoder input with the <SOS> token
        decoder_input = torch.tensor([[word2idx[SOS_TOKEN]]], dtype=torch.long).to(device)

        decoded_words = []

        for _ in range(max_length):
            decoder_embedded = model.decoder(decoder_input)
            output, hidden = model.decoder_rnn(decoder_embedded, hidden)
            output = model.fc_out(output)
            output = torch.softmax(output, dim=2)

            # Get the highest predicted word token from the output
            topv, topi = output.topk(1)
            next_word_idx = topi.squeeze().item()

            if next_word_idx == word2idx[EOS_TOKEN]:
                break
            else:
                decoded_words.append(idx2word.get(next_word_idx, UNK_TOKEN))

            # Prepare the next input for the decoder
            decoder_input = topi.squeeze().detach().unsqueeze(0)

        return ' '.join(decoded_words)

# Get chatbot response
def get_response(msg):
    response = generate_response(model, msg, word2idx, idx2word)
    if response:
        return response
    else:
        return "I didn't understand that. Can you rephrase?"

# Main loop to run the chatbot
if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit")
    while True:
        try:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                print("Goodbye!")
                break
            response = get_response(sentence)
            print("AI:", response)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
