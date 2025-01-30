import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import TextProcessor, TransformerChat
import os
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.serialization import add_safe_globals
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

class Config:
    batch_size = 64
    seq_length = 50
    learning_rate = 0.0002
    epochs = 200
    data_path = "data/chatbot_data.csv"
    checkpoint_path = "data/checkpoint.pth"
    best_model_path = "data/best_model.pth"
    dropout = 0.2

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
            padded = seq[:self.seq_length] + [0]*(self.seq_length - len(seq))
            return torch.LongTensor(padded)

        return {
            'question': process_text(question),
            'answer': process_text(answer)
        }

def train():
    # Load data
    df = pd.read_csv(Config.data_path)
    
    # Initialize processor
    processor = TextProcessor()
    processor.build_vocab(pd.concat([df['Question'], df['Answer']]))
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = ChatDataset(train_df['Question'], train_df['Answer'], processor)
    val_dataset = ChatDataset(val_df['Question'], val_df['Answer'], processor)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerChat(
        processor.vocab_size,
        dropout=Config.dropout
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.learning_rate,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=5
    )
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # Load checkpoint
    start_epoch = 0
    if os.path.exists(Config.checkpoint_path):
        try:
            add_safe_globals([defaultdict])
            checkpoint = torch.load(
                Config.checkpoint_path,
                map_location=device,
                weights_only=False
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            processor.__setstate__(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            start_epoch = 0

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    no_improvement = 0
    
    for epoch in range(start_epoch, Config.epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            src = batch['question'].to(device)
            tgt = batch['answer'].to(device)
            
            outputs = model(src, tgt[:, :-1])
            loss = criterion(outputs.view(-1, processor.vocab_size), tgt[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['question'].to(device)
                tgt = batch['answer'].to(device)
                outputs = model(src, tgt[:, :-1])
                val_loss += criterion(outputs.view(-1, processor.vocab_size), tgt[:, 1:].reshape(-1)).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break


        # Save checkpoint
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **processor.__getstate__()
        }
        torch.save(state, Config.checkpoint_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            torch.save(state, Config.best_model_path)

        logging.info(f"Epoch {epoch+1}/{Config.epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    train()