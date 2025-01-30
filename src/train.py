import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Define special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

# Hyperparameters
BATCH_SIZE = 16
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
TEACHER_FORCING_RATIO = 0.5
MAX_LENGTH = 20  # Maximum length of response

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load intents.json with explicit encoding to prevent UnicodeDecodeError
with open('data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Data Preparation
all_words = []
pairs = []

for intent in intents['intents']:
    patterns = intent['patterns']
    responses = intent['responses']
    for pattern in patterns:
        for response in responses:
            # Tokenize and lemmatize input pattern
            input_words = nltk.word_tokenize(pattern)
            input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words if word not in set(['?', '!', '.', ','])]
            
            # Tokenize and lemmatize response, then add <SOS> and <EOS>
            output_words = nltk.word_tokenize(response)
            output_words = [lemmatizer.lemmatize(word.lower()) for word in output_words if word not in set(['?', '!', '.', ','])]
            output_words = [SOS_TOKEN] + output_words + [EOS_TOKEN]
            
            # Extend the word list and add the pair
            all_words.extend(input_words)
            all_words.extend(output_words)
            pairs.append((input_words, output_words))

# Remove duplicates and sort the vocabulary
all_words = sorted(set(all_words))

# Create word2idx and idx2word mappings
word2idx = {word: idx for idx, word in enumerate([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + all_words)}
idx2word = {idx: word for word, idx in word2idx.items()}

vocab_size = len(word2idx)
print(f"Vocabulary Size: {vocab_size}")
print(f"Total Pairs: {len(pairs)}")

# Function to encode sequences
def encode_sequence(seq, word2idx):
    return [word2idx.get(word, word2idx[UNK_TOKEN]) for word in seq]

# Define the Dataset
class ChatDataset(Dataset):
    def __init__(self, pairs, word2idx, max_length=MAX_LENGTH):
        self.pairs = pairs
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, output_seq = self.pairs[idx]
        input_encoded = encode_sequence(input_seq, self.word2idx)
        output_encoded = encode_sequence(output_seq, self.word2idx)
        
        return torch.tensor(input_encoded, dtype=torch.long), torch.tensor(output_encoded, dtype=torch.long)

# Collate function for DataLoader to handle variable-length sequences
def collate_fn(batch):
    inputs, outputs = zip(*batch)
    input_lengths = [len(seq) for seq in inputs]
    output_lengths = [len(seq) for seq in outputs]
    
    # Pad sequences
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=word2idx[PAD_TOKEN])
    outputs_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=word2idx[PAD_TOKEN])
    
    return inputs_padded, outputs_padded, input_lengths, output_lengths

# Create Dataset and DataLoader
dataset = ChatDataset(pairs, word2idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=word2idx[PAD_TOKEN])
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, lengths):
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_size)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed)  # outputs shape: (batch_size, seq_length, hidden_size * 2)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        # Sum bidirectional hidden states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        
        return outputs, hidden  # outputs: (batch_size, seq_length, hidden_size), hidden: (num_layers, batch_size, hidden_size)

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=word2idx[PAD_TOKEN])
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, hidden):
        embedded = self.embedding(x)  # Shape: (batch_size, 1, embedding_size)
        output, hidden = self.gru(embedded, hidden)  # output: (batch_size, 1, hidden_size)
        predictions = self.fc_out(output)  # predictions: (batch_size, 1, output_size)
        predictions = self.softmax(predictions)  # predictions: (batch_size, 1, output_size)
        return predictions, hidden

# Define the Seq2Seq Model
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # Initialize tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0].unsqueeze(1)  # Shape: (batch_size, 1)
        
        for t in range(1, trg_len):
            # Pass through decoder
            output, hidden = self.decoder(input, hidden)  # output: (batch_size, 1, output_size)
            outputs[:, t, :] = output.squeeze(1)
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)  # Shape: (batch_size, 1)
            
            # Determine next input
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs  # Shape: (batch_size, trg_len, output_size)

# Instantiate the model
encoder = Encoder(input_size=vocab_size, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
decoder = Decoder(output_size=vocab_size, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
model = Seq2SeqModel(encoder, decoder, device).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    
    for inputs, outputs, input_lengths, output_lengths in progress:
        inputs, outputs = inputs.to(device), outputs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass, passing the input lengths
        output = model(
            src=inputs,
            src_lengths=input_lengths,
            trg=outputs,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )
        
        # Reshape for loss calculation
        # Exclude the first token (<SOS>) for both predictions and targets
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Shape: (batch_size * (trg_len -1), output_dim)
        outputs = outputs[:, 1:].reshape(-1)  # Shape: (batch_size * (trg_len -1))
        
        # Calculate loss
        loss = criterion(output, outputs)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
    
    # Save the model every 10 epochs
    if epoch % 10 == 0:
        torch.save({
            'model_state': model.state_dict(),
            'word2idx': word2idx,
            'idx2word': idx2word,
            'all_words': all_words
        }, f"seq2seq_chatbot_epoch{epoch}.pth")

# Save the final model after training completes
torch.save({
    'model_state': model.state_dict(),
    'word2idx': word2idx,
    'idx2word': idx2word,
    'all_words': all_words
}, "seq2seq_chatbot.pth")

print("Training complete and model saved as 'seq2seq_chatbot_final.pth'")
