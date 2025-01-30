import json
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

# --- Assistant Class Definition ---
class Assistant:
    def __init__(self, model_path, device=None):
        """
        Initializes the Assistant with the trained model and tokenizer.

        Args:
            model_path (str): Path to the directory containing the trained model.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to automatic detection.
        """
        # Initialize spaCy for entity extraction
        self.nlp = spacy.load("en_core_web_sm")

        # Device configuration
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add a unique pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        
        self.tokenizer.pad_token = '<PAD>'

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        
        # Resize token embeddings to accommodate the new pad token
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load PEFT (LoRA) model
        self.model = PeftModel.from_pretrained(self.model, model_path).to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Memory and context
        self.long_term_memory = {}
        self.conversation_history = []
        self.max_history_length = 5

    def update_memory(self, text):
        """
        Extracts entities from the text and updates the long-term memory.

        Args:
            text (str): Input text to extract entities from.
        """
        doc = self.nlp(text)
        for ent in doc.ents:
            label = ent.label_
            if label not in self.long_term_memory:
                self.long_term_memory[label] = []
            if ent.text not in self.long_term_memory[label]:
                self.long_term_memory[label].append(ent.text)

    def format_prompt(self, user_input):
        """
        Formats the prompt by combining memory, history, and current user input.

        Args:
            user_input (str): The latest input from the user.

        Returns:
            str: The formatted prompt.
        """
        memory_str = json.dumps(self.long_term_memory, indent=2)[:2000]  # Adjusted length for better context
        history = "\n".join(self.conversation_history[-self.max_history_length:])
        prompt = f"""
[Memory]
{memory_str}

[History]
{history}

[User]
{user_input}

[Bot]"""
        return prompt.strip()

    def generate_response(self, user_input, max_length=256, temperature=0.8, top_p=0.95, repetition_penalty=1.1):
        """
        Generates a response from the model based on the user input.

        Args:
            user_input (str): The latest input from the user.
            max_length (int, optional): Maximum length of the generated response. Defaults to 256.
            temperature (float, optional): Sampling temperature. Defaults to 0.8.
            top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
            repetition_penalty (float, optional): Penalty for repetition. Defaults to 1.1.

        Returns:
            str: The generated bot response.
        """
        # Update conversation context
        self.conversation_history.append(f"User: {user_input}")
        self.update_memory(user_input)

        # Generate prompt
        prompt = self.format_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Pass attention mask
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,  # Use the new pad token ID
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

        # Decode and process response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_response = full_response.split("[Bot]")[-1].strip()
        self.conversation_history.append(f"Bot: {bot_response}")

        return bot_response

# --- Main Chat Function ---
def main():
    parser = argparse.ArgumentParser(description="Chat with your trained AI assistant.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./chatbot_model",
        help="Path to the trained model directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on ('cuda' or 'cpu'). Defaults to automatic detection."
    )
    args = parser.parse_args()

    # Initialize Assistant
    assistant = Assistant(model_path=args.model_path, device=args.device)

    print("Welcome to your AI Assistant! Type 'exit', 'quit', or 'bye' to end the conversation.\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Bot: Goodbye! Have a great day!")
                break
            elif user_input == "":
                print("Bot: I'm here whenever you're ready to chat.")
                continue

            # Generate and print response
            response = assistant.generate_response(user_input)
            print(f"Bot: {response}\n")

        except KeyboardInterrupt:
            print("\nBot: Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
