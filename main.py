import subprocess
import os
import threading
import itertools
import time
import sys

def run_ollama(prompt, model="llama3"):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    output, error = process.communicate(input=prompt)

    if error:
        print("Error:", error)
    return output.strip()

def load_knowledge(base_dir="knowledge"):
    knowledge = {}
    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(base_dir, filename), "r") as file:
                    knowledge[filename] = file.read()
    return "\n\n".join(knowledge.values())


def show_loading(stop_event):
    for c in itertools.cycle(["⏳", "⌛", "💭", "🤔"]):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\rLoading {c}")
        sys.stdout.flush()
        time.sleep(0.2)
    sys.stdout.write("\r Request complete! ")

if __name__ == "__main__":
    print("Local LLM Chatbot - starting...")
    knowledge = load_knowledge()

    while True:
        user_input = input("\nYou: ")

        prompt = f"""
        Відповідай українською мовою. Якщо не знаєш відповіді, скажи "Не знаю". Ти локальний чат-бот.
        Використовуй наступні знання для відповіді на питання: {knowledge}
        Питання користувача: {user_input}
        """

        # loading
        stop_event = threading.Event()
        spinner = threading.Thread(target=show_loading, args=(stop_event,))
        spinner.start()

        # start Ollama
        response = run_ollama(prompt)

        # stop loading
        stop_event.set()
        spinner.join()

        print(f"\nBot: {response}")