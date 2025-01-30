import torch
import torch.nn as nn
import re
import math
from collections import defaultdict
import random

class TextProcessor:
    def __init__(self):
        self.START_TOKEN = "<SOS>"
        self.END_TOKEN = "<EOS>"
        self.word2idx = defaultdict(self._default_factory)
        self.idx2word = {
            0: "<PAD>",
            1: "<UNK>",
            2: self.START_TOKEN,
            3: self.END_TOKEN
        }
        self.vocab_size = 4
        self.clean_regex = re.compile(r"[^a-zA-Z0-9\s\.\?\!',:;-]")
        self.punct_regex = re.compile(r"([.!?,])")
        
    def _default_factory(self): 
        return 1
        
    def __getstate__(self):
        return {
            'START_TOKEN': self.START_TOKEN,
            'END_TOKEN': self.END_TOKEN,
            'word2idx': dict(self.word2idx),
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'clean_regex': self.clean_regex.pattern,
            'punct_regex': self.punct_regex.pattern
        }
        
    def __setstate__(self, state):
        self.START_TOKEN = state['START_TOKEN']
        self.END_TOKEN = state['END_TOKEN']
        self.word2idx = defaultdict(self._default_factory, state['word2idx'])
        self.idx2word = state['idx2word']
        self.vocab_size = state['vocab_size']
        self.clean_regex = re.compile(state['clean_regex'])
        self.punct_regex = re.compile(state['punct_regex'])

    def clean_text(self, text):
        text = str(text)
        text = self.punct_regex.sub(r" \1 ", text)
        text = self.clean_regex.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def build_vocab(self, texts):
        word_counts = defaultdict(int)
        for text in texts:
            cleaned = self.clean_text(text)
            for word in cleaned.split():
                word_counts[word] += 1

        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for idx, (word, _) in enumerate(sorted_words[:15000-4]):
            self.word2idx[word] = idx + 4
            self.idx2word[idx+4] = word
            self.vocab_size += 1

    def text_to_sequence(self, text):
        cleaned = self.clean_text(text)
        return [self.word2idx[self.START_TOKEN]] + \
               [self.word2idx.get(word, 1) for word in cleaned.split()] + \
               [self.word2idx[self.END_TOKEN]]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TransformerChat(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        src = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.fc(self.dropout(output))