import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===============================
# ⚙️ 1. Конфігурація
# ===============================
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   # базова модель
ADAPTER_PATH = "saved_models"                          # твоя збережена LoRA модель

# ===============================
# 🔤 2. Завантаження токенайзера
# ===============================
print("🔤 Завантаження токенайзера...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ===============================
# 🧩 3. Завантаження базової моделі
# ===============================
print("🧠 Завантаження базової моделі...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# ===============================
# 🧩 4. Завантаження LoRA адаптера
# ===============================
print("🔗 Підключення LoRA адаптера...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ===============================
# 💬 5. Генерація тексту
# ===============================
prompt = "В якій сім'ї жив Поттер?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n🚀 Генерація відповіді...\n")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print("🧾 Відповідь:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
