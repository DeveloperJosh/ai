from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import json
import spacy
from peft import LoraConfig, get_peft_model

# Initialize spaCy and tokenizer first
nlp = spacy.load("en_core_web_sm")
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --- Enhanced Dataset with Memory Context ---
expanded_dataset = [
    # Original dataset entries
    {
        "input": "Hello",
        "response": "Hi! How can I assist you today?",
        "memory": {}
    },
    # ... (all previous entries)
    # New memory-aware examples
    {
        "input": "My sister Emily is coming over tonight",
        "response": "Should I prepare Emily's favorite lasagna?",
        "memory": {"family": {"sister": "Emily", "favorite_food": "lasagna"}}
    },
    {
        "input": "What's my car's color?",
        "response": "Your Tesla Model 3 is blue, just like your eyes.",
        "memory": {"car": {"model": "Tesla Model 3", "color": "blue"}}
    },
    # ... (50+ additional examples with memory context)
]

# Save and load dataset
with open("enhanced_dataset.json", "w") as f:
    json.dump(expanded_dataset, f)

dataset = load_dataset("json", data_files="enhanced_dataset.json")["train"]
dataset = dataset.train_test_split(test_size=0.15)

# --- Data Processing ---
def preprocess_function(examples):
    formatted_texts = []
    for i in range(len(examples["input"])):
        memory_str = json.dumps(examples["memory"][i])[:200]
        text = f"""
        [Memory] {memory_str}
        [User] {examples['input'][i]}
        [Bot] {examples['response'][i]}{tokenizer.eos_token}"""
        formatted_texts.append(text.strip())
    
    return tokenizer(
        formatted_texts,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# --- Memory-Augmented Model Architecture ---
class Assistant:
    def __init__(self):
        # Model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = tokenizer
        
        # Initialize LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.1,
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Memory and context
        self.long_term_memory = {}
        self.conversation_history = []
        self.max_history_length = 5
        
    def update_memory(self, text):
        # Advanced entity extraction with spaCy
        doc = nlp(text)
        for ent in doc.ents:
            label = ent.label_
            if label not in self.long_term_memory:
                self.long_term_memory[label] = []
            if ent.text not in self.long_term_memory[label]:
                self.long_term_memory[label].append(ent.text)
                
    def format_prompt(self, user_input):
        # Combine memory, history, and current input
        memory_str = json.dumps(self.long_term_memory)[:200]
        history = "\n".join(self.conversation_history[-self.max_history_length:])
        return f"""
        [Memory] {memory_str}
        [History] {history}
        [User] {user_input}
        [Bot]""".strip()

    def generate_response(self, user_input):
        # Update conversation context
        self.conversation_history.append(f"User: {user_input}")
        self.update_memory(user_input)
        
        # Generate response
        prompt = self.format_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=256,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        # Process response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_response = full_response.split("[Bot]")[-1].strip()
        self.conversation_history.append(f"Bot: {bot_response}")
        
        return bot_response

# --- Training Setup ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./romantic_assistant_v1",
    num_train_epochs=20,
    per_device_train_batch_size=8,  # Adjusted for stability
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_steps=100,
    fp16=True,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    deepspeed="ds_config.json"
)

assistant = Assistant()
trainer = Trainer(
    model=assistant.model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)

# Start training
trainer.train()

# Save final model
assistant.model.save_pretrained("./chatbot_model")
assistant.tokenizer.save_pretrained("./chatbot_model")