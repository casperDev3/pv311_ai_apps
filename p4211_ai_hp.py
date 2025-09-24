# Завдання:
# Написати модель, що даватиме відповіді на
# запитання по книзі Гарі Поттер

import torch
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
from torch.utils.data import DataLoader, Dataset
import nltk
from nltk.tokenize import sent_tokenize
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt_tab', quiet=True)


class HarryPotterQADataset(Dataset):
    def __init__(self, text_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Перевіряємо чи існує файл
        if not os.path.exists(text_file):
            print(f"Файл {text_file} не знайдено. Створюємо тестовий текст...")
            self.text = self.create_sample_text()
        else:
            with open(text_file, 'r', encoding='utf-8') as file:
                self.text = file.read()

        self.sentences = sent_tokenize(self.text)

    def create_sample_text(self):
        """Створює тестовий текст про Гарі Поттера"""
        return """
        Harry Potter is a young wizard who lives with his aunt and uncle. 
        His best friend is Ron Weasley, and he also has a close friend named Hermione Granger. 
        Harry attends Hogwarts School of Witchcraft and Wizardry. 
        He has a pet owl named Hedwig. 
        Harry's nemesis is Lord Voldemort, a dark wizard. 
        Harry plays Quidditch as a Seeker for Gryffindor house. 
        His parents, James and Lily Potter, were killed by Voldemort when Harry was a baby.
        Albus Dumbledore is the headmaster of Hogwarts and Harry's mentor.
        Harry has a lightning bolt scar on his forehead.
        """

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten()
        }


def find_best_context(question, sentences, max_sentences=5):
    """Знаходить найкращий контекст для запитання"""
    # Словник ключових слів для різних типів питань
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

    # Визначаємо тип питання та відповідні ключові слова
    for key, words in keywords_map.items():
        if any(word in question_lower for word in key.split()):
            relevant_keywords.extend(words)

    # Якщо не знайшли спеціальних ключових слів, використовуємо слова з питання
    if not relevant_keywords:
        # Фільтруємо службові слова
        stop_words = {"who", "what", "where", "when", "why", "how", "is", "are", "was", "were", "the", "a", "an"}
        relevant_keywords = [word.lower() for word in question.split() if word.lower() not in stop_words]

    sentence_scores = []

    for i, sentence in enumerate(sentences):
        score = 0
        sentence_lower = sentence.lower()

        # Підраховуємо релевантність
        for keyword in relevant_keywords:
            if keyword in sentence_lower:
                # Бонус за точне співпадіння
                score += 2 if keyword in sentence_lower.split() else 1

        sentence_scores.append((score, i, sentence))

    # Сортуємо за релевантністю
    sentence_scores.sort(key=lambda x: x[0], reverse=True)

    # Беремо топ найкращих речень, але тільки ті що мають score > 0
    best_sentences = []
    for score, idx, sentence in sentence_scores[:max_sentences * 2]:  # беремо більше для фільтрації
        if score > 0:
            best_sentences.append(sentence)
            if len(best_sentences) >= max_sentences:
                break

    # Якщо не знайшли релевантних речень, беремо перші кілька
    if not best_sentences:
        best_sentences = sentences[:max_sentences]


def find_best_context_tfidf(question, sentences, max_sentences=3):
    """Використовує TF-IDF для знаходження найрелевантніших речень"""
    # Очищаємо речення від зайвих символів
    clean_sentences = []
    for sentence in sentences:
        # Видаляємо зайві пробіли та символи
        clean_sentence = re.sub(r'\s+', ' ', sentence.strip())
        if len(clean_sentence) > 20:  # Фільтруємо дуже короткі речення
            clean_sentences.append(clean_sentence)

    if not clean_sentences:
        return sentences[0] if sentences else ""

    # Створюємо TF-IDF векторизатор
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)  # Використовуємо униграми та біграми
    )

    try:
        # Векторизуємо речення
        all_texts = clean_sentences + [question]
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Обчислюємо схожість між питанням та реченнями
        question_vector = tfidf_matrix[-1]  # Останній вектор - це питання
        sentence_vectors = tfidf_matrix[:-1]  # Всі інші - речення

        similarities = cosine_similarity(question_vector, sentence_vectors).flatten()

        # Знаходимо найбільш схожі речення
        top_indices = np.argsort(similarities)[::-1][:max_sentences]

        best_sentences = [clean_sentences[i] for i in top_indices if similarities[i] > 0.01]

        if not best_sentences:
            # Якщо TF-IDF не дав результатів, використовуємо ключові слова
            return find_best_context(question, clean_sentences, max_sentences)

        return " ".join(best_sentences)

    except Exception as e:
        print(f"TF-IDF пошук не вдався: {e}")
        return find_best_context(question, clean_sentences, max_sentences)


def answer_question_simple(model, tokenizer, question, context, device):
    """Проста функція для отримання відповіді"""
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1

    # Перевіряємо чи індекси валідні
    if start_idx >= end_idx or start_idx == 0:
        return "Не вдалося знайти відповідь"

    answer_tokens = input_ids[0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer if answer.strip() else "Не вдалося знайти відповідь"


def answer_question_pipeline(question, context):
    """Використовує готовий pipeline для QA"""
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        tokenizer="distilbert-base-cased-distilled-squad"
    )

    result = qa_pipeline(question=question, context=context)
    return result['answer']


if __name__ == "__main__":
    print("Starting QA model...")

    # Визначення пристрою
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Створюємо датасет
    print("Preparing dataset...")
    text_file = "data/harry_potter_book.txt"

    # Для демонстрації використаємо готову модель
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = HarryPotterQADataset(text_file, tokenizer)

    print(f"Loaded {len(dataset.sentences)} sentences from text")

    # Приклади запитань
    questions = [
        "Who is Harry Potter's best friend?",
        "What is the name of Harry's owl?",
        "Who is the headmaster of Hogwarts?",
        "What house does Harry play Quidditch for?",
        "What killed Harry's parents?"
    ]

    print("\n" + "=" * 50)
    print("ВІДПОВІДІ НА ЗАПИТАННЯ")
    print("=" * 50)

    # Відповідаємо на запитання використовуючи pipeline (рекомендований підхід)
    for question in questions:
        print(f"\nПитання: {question}")

        # Використовуємо покращений пошук контексту
        context = find_best_context_tfidf(question, dataset.sentences, max_sentences=len(dataset.sentences))

        # Обмежуємо довжину контексту для читабельності
        context_preview = context[:200] + "..." if len(context) > 200 else context
        print(f"Контекст: {context_preview}")

        try:
            # Використовуємо готову навчену модель
            answer = answer_question_pipeline(question, context)
            print(f"Відповідь: {answer}")
        except Exception as e:
            print(f"Помилка: {e}")
            print("Відповідь: Не вдалося знайти відповідь")

        print("-" * 30)

    print("\nМодель завершила роботу успішно!")