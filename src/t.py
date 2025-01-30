import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import os
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.serialization import add_safe_globals
from collections import defaultdict
from model import TextProcessor, TransformerChat
#oka

###############################################################################
#                             CONFIGURATION                                   #
###############################################################################
logging.basicConfig(level=logging.INFO)

class Config:
    # File system and training config
    data_path = "data/intents.json"     # Path to your JSON file with "intents"
    checkpoint_path = "data/checkpoint.pth"
    best_model_path = "data/best_model.pth"
    
    # Hyperparameters
    batch_size = 64
    seq_length = 50
    learning_rate = 0.0001
    epochs = 10
    dropout = 0.1

###############################################################################
#                            MODEL & DATASET                                  #
###############################################################################

# You need to have a model.py (or similar) with:
#   class TextProcessor: (for tokenization, building vocab, etc.)
#   class TransformerChat: (your model definition)
#
# If you do not have these, see the reference code below the main script.

# ----------- For reference only (assuming you have these in model.py) ----------
# from model import TextProcessor, TransformerChat


class ChatDataset(Dataset):
    def __init__(self, questions, answers, processor):
        self.processor = processor
        self.pairs = list(zip(questions, answers))
        self.seq_length = Config.seq_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]

        def process_text(text):
            seq = self.processor.text_to_sequence(text)
            # Pad or truncate to fixed sequence length
            padded = seq[:self.seq_length] + [0]*(self.seq_length - len(seq))
            return torch.LongTensor(padded)

        return {
            'question': process_text(question),
            'answer': process_text(answer)
        }

###############################################################################
#                              TRAIN FUNCTION                                 #
###############################################################################
def train():
    # -------------------------------------------------------------------------
    # 1. Load the JSON file and flatten it into (Question, Answer) pairs
    # -------------------------------------------------------------------------
    with open(Config.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # data should contain something like {"intents": [...]}

    all_pairs = []
    for intent in data["intents"]:
        patterns = intent["patterns"]
        responses = intent["responses"]
        # Create pairs: each pattern matched with each response (Cartesian product)
        for pat in patterns:
            for resp in responses:
                all_pairs.append((pat, resp))

    # Turn these pairs into a DataFrame
    df = pd.DataFrame(all_pairs, columns=["Question", "Answer"])

    # -------------------------------------------------------------------------
    # 2. Initialize processor & build vocab
    # -------------------------------------------------------------------------
    processor = TextProcessor()
    processor.build_vocab(pd.concat([df["Question"], df["Answer"]]))

    # -------------------------------------------------------------------------
    # 3. Train-test split
    # -------------------------------------------------------------------------
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # -------------------------------------------------------------------------
    # 4. Create datasets & loaders
    # -------------------------------------------------------------------------
    train_dataset = ChatDataset(train_df["Question"], train_df["Answer"], processor)
    val_dataset = ChatDataset(val_df["Question"], val_df["Answer"], processor)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

    # -------------------------------------------------------------------------
    # 5. Model setup
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerChat(
        processor.vocab_size,
        dropout=Config.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.learning_rate,
        weight_decay=0.01
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # CrossEntropy with label smoothing, ignoring padding index 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # -------------------------------------------------------------------------
    # 6. (Optional) Load from checkpoint if available
    # -------------------------------------------------------------------------
    start_epoch = 0
    if os.path.exists(Config.checkpoint_path):
        try:
            add_safe_globals([defaultdict])
            checkpoint = torch.load(Config.checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            processor.__setstate__(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            start_epoch = 0

    # -------------------------------------------------------------------------
    # 7. Training loop
    # -------------------------------------------------------------------------
    best_val_loss = float('inf')

    for epoch in range(start_epoch, Config.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            src = batch['question'].to(device)
            tgt = batch['answer'].to(device)
            
            # Generate predictions using teacher forcing for training
            outputs = model(src, tgt[:, :-1])  # shift target by 1
            loss = criterion(
                outputs.view(-1, processor.vocab_size), 
                tgt[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['question'].to(device)
                tgt = batch['answer'].to(device)
                outputs = model(src, tgt[:, :-1])
                val_loss += criterion(
                    outputs.view(-1, processor.vocab_size), 
                    tgt[:, 1:].reshape(-1)
                ).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Save checkpoint each epoch
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **processor.__getstate__()  # Save processor state
        }
        torch.save(state, Config.checkpoint_path)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(state, Config.best_model_path)

        logging.info(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

###############################################################################
#                                  MAIN                                       #
###############################################################################
if __name__ == "__main__":
    train()
