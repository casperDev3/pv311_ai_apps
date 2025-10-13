import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model # Використання PEFT для LoRA адаптації
from datasets import Dataset

# Конфігурація
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
BOOK_PATH = "data/book.txt"
OUTPUT_DIR = "saved_models"

# Підготовка даних
if not os.path.exists(BOOK_PATH):
    raise FileNotFoundError(f"File {BOOK_PATH} not found.")

with open(BOOK_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Розбивка тексту на частини
chunks = []
chunk_size = 800
for i in range(0, len(text), chunk_size):
    part = text[i:i+chunk_size]
    chunks.append({
        "instruction": "Поясни коротко цей текст:",
        "output": part
    })

dataset = Dataset.from_list(chunks)

# Завантаження моделі та токенізатора
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    prompt = [
        f"### Instruction: {inst}\n\nOutput: {out}" for inst, out in zip(batch["instruction"], batch["output"])
    ]
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

lora_config = LoraConfig(
    r=16, # Ранг адаптації
    lora_alpha=32, # Шкала адаптації
    target_modules=["q_proj", "v_proj"], # Цільові модулі для LoRA
    lora_dropout=0.05, # Дропаут для LoRA
    bias='none', # Без зміщення
    task_type="CAUSAL_LM" # Тип задачі
)

model = get_peft_model(model, lora_config)

# Налаштування тренування
print("Setting up training arguments...")
args = TrainingArguments(
    per_device_train_batch_size=1, # Розмір батчу
    gradient_accumulation_steps=4, # Кроки акумуляції градієнтів
    warmup_steps=5, # Кроки розігріву
    max_steps=50, # Максимальні кроки тренування
    learning_rate=2e-4, # Швидкість навчання
    fp16=torch.cuda.is_available(), # Використання FP16, якщо є GPU
    logging_steps=10, # Кроки логування
    output_dir=OUTPUT_DIR, # Директорія виводу
    save_total_limit=2, # Ліміт збережень
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)
print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)

print("Training complete. Saved to", OUTPUT_DIR)

print("Use trained model for inference:")
prompt = "В якій сім'ї народився проживав Гаррі Поттер?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    print("Hello, World!")