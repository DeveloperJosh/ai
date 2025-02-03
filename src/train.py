import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ===============================
# CONFIGURATION AND INITIAL SETUP
# ===============================

# Path to your fine-tuned GPT-2 model directory or a model name (e.g., "gpt2").
# If you're using the base GPT-2 from the Hugging Face Hub, you can set MODEL_PATH = "gpt2"
MODEL_PATH = "gpt2"  # Change as needed

# File to log conversations for offline learning/fine-tuning.
LOG_FILE = "conversation_logs.json"

# Set device: use GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model.
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Ensure the tokenizer has a pad token. GPT-2 doesn't have one by default.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===============================
# SET UP THE BOT'S PERSONALITY & HISTORY
# ===============================

# Define the bot's persona (e.g., from Persona-Chat).
persona = [
    "I am friendly and talkative.",
    "I love discussing interesting topics.",
    "I always try to understand the user."
]

# Initialize conversation history with the bot's persona.
conversation_history = [f"Persona: {' '.join(persona)}"]

# ===============================
# FUNCTION DEFINITIONS
# ===============================

def generate_response(user_input, history, max_history_turns=10):
    """
    Appends the user's input to the conversation history, generates a response using GPT-2,
    and then appends the bot's reply to the history.

    Args:
        user_input (str): The input string from the user.
        history (list): List of conversation turns.
        max_history_turns (int): Maximum number of turns (lines) to include in the prompt.

    Returns:
        response (str): The generated response from the bot.
        history (list): Updated conversation history including the bot's response.
    """
    # Append user's message.
    history.append("User: " + user_input)

    # Build the prompt using the most recent conversation turns.
    prompt = "\n".join(history[-max_history_turns:]) + "\nBot:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Create an attention mask: ones for each token (since we're not padding within this prompt).
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Determine maximum length for generation (keeping GPT-2's context window in mind).
    max_length = min(1024, input_ids.shape[1] + 100)

    # Generate the response using the attention mask.
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,    # Enable sampling for creative responses.
        top_k=50,          # Use top-k sampling.
        top_p=0.95,        # Use nucleus (top-p) sampling.
        temperature=0.7    # Adjust temperature for randomness.
    )

    # Decode the generated output.
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract the bot's response from the generated text.
    if "Bot:" in output_text:
        response = output_text.split("Bot:")[-1].strip()
    else:
        response = output_text.strip()

    # Append the bot's response to the conversation history.
    history.append("Bot: " + response)
    return response, history

def save_conversation_log(history, filename=LOG_FILE):
    """
    Saves the conversation history to a JSON file.
    Each conversation is stored as a separate JSON object on a new line.

    Args:
        history (list): The conversation history to save.
        filename (str): The filename where the conversation log is stored.
    """
    log_entry = {"conversation": history}
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(log_entry, f)
        f.write("\n")
    print(f"Conversation log saved to {filename}.")

def load_previous_conversations(filename=LOG_FILE):
    """
    Loads previous conversation logs from the specified file.
    This can be useful for offline analysis or further fine-tuning.

    Args:
        filename (str): The filename to load logs from.

    Returns:
        conversations (list): A list of conversation logs.
    """
    if not os.path.exists(filename):
        return []
    conversations = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                conversations.append(entry)
            except json.JSONDecodeError:
                continue
    return conversations

# ===============================
# MAIN INTERACTIVE CHAT LOOP
# ===============================

def main():
    print("Chat with the bot! Type 'quit' or 'exit' to end the conversation.")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting conversation. Saving conversation log...")
                save_conversation_log(conversation_history)
                break
            response, updated_history = generate_response(user_input, conversation_history)
            print("Bot:", response)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting and saving conversation log...")
        save_conversation_log(conversation_history)

if __name__ == "__main__":
    main()
