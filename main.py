# Завдання:
# Написати модель, що даватиме відповіді на
# запитання по книзі Гарі Поттер

import torch
from sympy import vectorize
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
from torch.utils.data import DataLoader, Dataset
import nltk # Для обробки тексту
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt_tab')

class HarryPotterQADataset(Dataset):
    def __init__(self, text, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len # Максимальна довжина вхідного тексту
        with open(text, 'r', encoding='utf-8') as file:
            self.text = file.read()
        self.sentences  = sent_tokenize(self.text)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True, # Додаємо спеціальні токени [CLS] і [SEP]
            max_length=self.max_len, # Максимальна довжина
            truncation=True, # Обрізаємо, якщо довше за max_len
            padding='max_length', # Доповнюємо до max_len
            return_tensors='pt' # Повертаємо тензори PyTorch
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten()
        }

# Функція для отримання відповіді на запитання
def answer_question(model, tokenizer, question, context, device):
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        tokenizer="distilbert-base-cased-distilled-squad"
    )
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def find_best_context(question, sentences, max_sentences=3):
    keywords_map = {
        "best friend": ["ron", "weasley", "hermione", "granger", "friend"],
        "owl": ["hedwig", "owl", "bird"],
        "headmaster": ["dumbledore", "albus", "headmaster", "principal"],
        "quidditch": ["quidditch", "gryffindor", "seeker", "house", "team"],
        "killed": ["voldemort", "curse", "avada kedavra", "murdered", "killed"],
        "parents": ["james", "lily", "potter", "mother", "father", "parents"]
    }

    question_lower = question.lower()
    relevant_keywords = []

    for key, words in keywords_map.items():
        if any(word in question_lower for word in key.split()):
            relevant_keywords.extend(words)

    if not relevant_keywords:
        stop_words = {"who", "what", "where", "when", "why", "how", "is", "are", "was", "were", "the", "a", "an"}
        relevant_keywords = [word.lower() for word in question.split() if word.lower() not in stop_words]

    sentence_scores  = []

    for i, sentence in enumerate(sentences):
        score = 0
        sentence_lower = sentence.lower()

        for keyword in relevant_keywords:
            if keyword in sentence_lower:
                score += 2 if keyword in sentence_lower.split() else 1

        if score > 0:
            sentence_scores.append((i, score))

    sentence_scores.sort(key= lambda x: x[0], reverse=True)

    best_sentences = [sentences[i] for i, score in sentence_scores[:max_sentences]]
    return " ".join(best_sentences) if best_sentences else sentences[0] if sentences else ""




def find_best_context_tfidf(question, sentences, max_sentences=3):
    clean_sentences = []
    for sentence in sentences:
        clean_sentence = sentence.replace('\n', ' ').strip()
        if clean_sentence:
            clean_sentences.append(clean_sentence)

    if not clean_sentences:
        return sentences[0] if sentences else ""

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000, # Максимальна кількість ознак
        ngram_range=(1, 2) # Використовуємо униграми та біграми
    )
    try:
        all_texts = clean_sentences + [question]
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        question_vector = tfidf_matrix[-1] # Останній вектор - це питання
        sentence_vectors = tfidf_matrix[:-1] # Всі інші - речення

        similarities = cosine_similarity(question_vector, sentence_vectors).flatten()

        top_indices = np.argsort(similarities)[::-1][:max_sentences] # Індекси найбільш схожих речень
        best_sentences = [clean_sentences[i] for i in top_indices if similarities[i] > 0.01] # Фільтруємо за порогом схожості

        if not best_sentences:
            return find_best_context(question, clean_sentences, max_sentences)
        return " ".join(best_sentences)
    except Exception as e:
        print(f"TF-IDF пошук не вдався: {e}")
        return find_best_context(question, clean_sentences, max_sentences)




if __name__ == "__main__":
    print("Starting QA model...")
   # Визначення пристрою (CPU або GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Завантажуємо модель BERT для відповіді на запитання
    print("Loading model...")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

   # Шлях до тексту книги Гарі Поттер
    text_file = "data/harry_potter_book.txt"

   # Створюємо датасет і завантажувач даних
    print("Preparing dataset...")
    dataset = HarryPotterQADataset(text_file, tokenizer)

   # Приклада запитання
    question = "Who is the headmaster of Hogwarts?"
    context = find_best_context_tfidf(question, dataset.sentences, max_sentences=len(dataset.sentences))  # Використовуємо перше речення як контекст

    print(context)
   # Отримуємо відповідь
    print("Answering question...")
    print("-" * 20)

    answer = answer_question(model, tokenizer, question, context, device)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

