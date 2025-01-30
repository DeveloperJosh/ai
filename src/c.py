import torch
from model import TextProcessor, TransformerChat
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

class ChatBot:
    def __init__(self, model_path='data/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.context = []
        self.max_context_length = 3  # Remember last 3 exchanges
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.processor = TextProcessor()
            self.processor.__setstate__(checkpoint)
            self.model = TransformerChat(
                self.processor.vocab_size,
                dropout=0.1  # Slight dropout for variation
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logging.info("Model loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def generate_response(self, input_text, max_length=50, temp=0.7, top_p=0.9):
        try:
            # Update context
            self._update_context(f"You: {input_text}")
            
            # Create context-aware input
            context = "\n".join(self.context[-self.max_context_length:])
            full_input = f"{context}\nAI:"
            
            input_seq = self.processor.text_to_sequence(full_input)
            src = torch.LongTensor([input_seq]).to(self.device)
            
            # Initialize generation
            generated = [self.processor.word2idx[self.processor.START_TOKEN]]
            for _ in range(max_length):
                tgt = torch.LongTensor([generated]).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(src, tgt)
                
                # Get next token probabilities
                logits = outputs[:, -1, :] / temp
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # Apply repetition penalty
                probs = self._apply_repetition_penalty(generated, probs, penalty=1.2)
                
                # Top-p sampling
                sorted_indices = np.argsort(-probs)
                cumulative = np.cumsum(probs[sorted_indices])
                mask = cumulative <= top_p
                mask[np.argmax(~mask)] = True  # Include first token exceeding p
                valid_indices = sorted_indices[mask]
                
                # Resample probabilities
                filtered_probs = probs[valid_indices]
                filtered_probs /= filtered_probs.sum()
                
                next_token = np.random.choice(valid_indices, p=filtered_probs)
                
                if next_token == self.processor.word2idx[self.processor.END_TOKEN]:
                    break
                    
                generated.append(next_token)
                
                # Prevent infinite loops
                if len(generated) > 5 and len(set(generated[-3:])) == 1:
                    break

            response = self._sequence_to_text(generated)
            self._update_context(f"AI: {response}")
            return response
            
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "Hmm, let me think about that differently..."

    def _apply_repetition_penalty(self, generated, probs, penalty=1.2):
        """Discourage repetitive outputs"""
        for token in set(generated[-4:]):  # Look at last 4 tokens
            try: probs[token] /= penalty
            except: pass
        probs /= probs.sum()  # Renormalize
        return probs

    def _update_context(self, text):
        """Manage conversation history"""
        self.context.append(text)
        if len(self.context) > self.max_context_length * 2:
            self.context = self.context[-self.max_context_length * 2:]

    def _sequence_to_text(self, sequence):
        """Convert tokens to text with better formatting"""
        words = []
        for idx in sequence:
            token = self.processor.idx2word.get(idx, '')
            if token in [self.processor.START_TOKEN, self.processor.END_TOKEN]:
                continue
            if token in ['.', '!', '?'] and words:
                words[-1] += token  # Attach punctuation to previous word
            else:
                words.append(token)
        return ' '.join(words).capitalize()

if __name__ == "__main__":
    bot = ChatBot()
    print("Chat with AI (type 'quit' to exit)")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            response = bot.generate_response(
                user_input,
                temp=5.0,  # Higher = more creative
                top_p=5.0  # Broader word selection
            )
            print(f"AI: {response}")
    except KeyboardInterrupt:
        print("\nGoodbye!")