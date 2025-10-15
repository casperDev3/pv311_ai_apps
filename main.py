import os
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv
from torch.ao.quantization.backend_config.onednn import with_bn

# setting up OpenAI client
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# generate text using OpenAI
def generate_text():
    topic = input("Enter a topic: ")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ти є помічником, який допомагає генерувати текст. Ти відповідаєш українською мовою."},
            {"role": "user", "content": f"Write a short paragraph about {topic}."}
        ]
    )
    text = response.choices[0].message.content
    print("\nGenerated Text:\n", text)
    return text

# generate image using OpenAI
def generate_image():
    prompt = input("Enter a prompt: ")
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        # n=1,
        # size="1024x1024"
    )
    url = response.data[0].url
    print("\nGenerated Image URL:\n", url)

def text_to_speech(text):
    # text = input("Enter text: ")
    tts = gTTS(text=text, lang='uk')
    filename = 'speech.mp3'
    tts.save(filename)
    print(f"Generated speech saved to {filename}")
    os.system(f"open {filename}")

def chat_mode():
    print("Entering chat mode. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ти є помічником, який допомагає генерувати текст. Ти відповідаєш українською мовою."},
                {"role": "user", "content": user_input}
            ]
        )
        text = response.choices[0].message.content
        print("AI:", text)


def main():
    print("Please choose an option:")
    print("1. Generate Text")
    print("2. Generate Image")
    print("3. Text to Speech")
    print("4. Chat Mode")
    print("5. Exit")

    while True:
        choice = input("Enter your choice (1-5): ")
        if choice == '1':
            generate_text()
        elif choice == '2':
            generate_image()
        elif choice == '3':
            text_to_speech(generate_text())
        elif choice == '4':
            chat_mode()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
    # print(API_KEY)