import os
import requests
import speech_recognition as sr # розпізнавання голосу
from gtts import gTTS # озвучка
import tempfile
import platform
import subprocess

# ---  Конфік ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or ""
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def play_audio(path):
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.run(["mpg123", path])
    except Exception as e:
        print(f"Error playing audio: {e}")

def listen_ukrainian(timeout = 5, phrase_time_limit=20):
    r = sr.Recognizer() # розпізнавач мови

    with sr.Microphone(device_index=10) as source:
        print("Говори щось українською!")
        print(sr.Microphone.list_microphone_names())
        r.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio =  r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("Нічого не почув!")
            return None
    try:
        text = r.recognize_google(audio, language="uk-UA")
        print("Ти скадав (ла): ", text)
        return text
    except Exception as err:
        print("Не вдалось розпізнати!")
        print("Ймовірна помилка: ", err)
    return None

def ask_groq(prompt = "test", system="Ти - розумний і доброзичливий у спілкування український асистент. Відповідай коротко і просто."):
     headers = {
         "Authorization": f"Bearer {GROQ_API_KEY}",
         "Content-Type": "application/json"
     }

     payload = {
         "model": MODEL,
         "messages": [
             {
                 "role": "system", "content": system
             },
             {
                 "role": "user", "content": prompt
             }
         ],
         "temperature": 0.7, # варіативність
         "max_tokens": 500,
         "stream": False
     }

     try:
         res = requests.post(GROQ_URL, headers=headers, json=payload)
         if res.status_code != 200:
             print("Сталась помилка!")
             print(f"Error: {res.status_code} {res.text}")
             return "Вибач сталась помилка! При запиті до ШІ!"
         data = res.json()
         return data["choices"][0]["message"]["content"]
     except Exception as err:
         print("Помилка при запиті", err)
         return "Вибач сталась помилка! При запиті до ШІ!"

if __name__ == "__main__":
    query = listen_ukrainian()
    answer = ask_groq(query)
    print(answer)