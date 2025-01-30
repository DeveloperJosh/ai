import torch
from model import TextProcessor, TransformerChat
import logging

logging.basicConfig(level=logging.INFO)

class ChatBot:
    def __init__(self, model_path='data/checkpoint.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
    def load_model(self, model_path):
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Rebuild processor
            self.processor = TextProcessor()
            self.processor.word2idx = checkpoint['word2idx']
            self.processor.idx2word = checkpoint['idx2word']
            self.processor.vocab_size = checkpoint['vocab_size']
            self.processor.START_TOKEN = checkpoint['START_TOKEN']
            self.processor.END_TOKEN = checkpoint['END_TOKEN']
            
            # Initialize model
            self.model = TransformerChat(self.processor.vocab_size).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logging.info("Model loaded successfully!")
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def generate_response(self, input_text, max_length=50, temp=0.7, top_p=0.9):
        try:
            # Process input
            input_seq = self.processor.text_to_sequence(input_text)
            src = torch.LongTensor([input_seq]).to(self.device)
            
            # Initialize output
            tgt = torch.LongTensor([[self.processor.word2idx[self.processor.START_TOKEN]]]).to(self.device)
            
            for _ in range(max_length):
                with torch.no_grad():
                    output = self.model(src, tgt)
                
                logits = output[:, -1, :] / temp
                probs = torch.softmax(logits, dim=-1)
                
                # Top-p sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs <= top_p
                valid_indices = sorted_indices[mask]
                
                if valid_indices.numel() == 0:
                    break
                
                next_token = torch.multinomial(sorted_probs[mask], 1)
                next_token = valid_indices[next_token]
                
                if next_token.item() == self.processor.word2idx[self.processor.END_TOKEN]:
                    break
                
                tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)

            return self._sequence_to_text(tgt[0])
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I'm having trouble thinking right now."

    def _sequence_to_text(self, sequence):
        words = []
        for idx in sequence:
            token = self.processor.idx2word.get(idx.item(), '')
            if token in [self.processor.START_TOKEN, self.processor.END_TOKEN]:
                continue
            words.append(token)
        return ' '.join(words).capitalize()

if __name__ == "__main__":
    try:
        bot = ChatBot()
        print("Chat with AI (type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            #print(user_input)
            response = bot.generate_response(user_input)
            print(f"AI: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")