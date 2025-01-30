from transformers import pipeline, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer from the saved directory
model_path = "./chatbot_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Load tokenizer
chatbot = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,  # Now properly defined
    device=0 if torch.cuda.is_available() else -1,
)

# Chat loop
print("Chatbot: Hi! How can I help you? (Type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # Generate response
    prompt = f"User: {user_input}\nBot:"
    response = chatbot(
        prompt,
        max_length=150,
        truncation=True,
        num_return_sequences=1,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  # Stop generation at EOS token
    )
    
    # Extract bot reply
    print(response)
    bot_reply = response[0]['generated_text'].split("Bot:")[-1].strip()
    print(f"Chatbot: {bot_reply}")