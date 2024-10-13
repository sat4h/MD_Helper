import requests
from sentence_transformers import util, SentenceTransformer
import pickle
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI  # Импортируем библиотеку для работы с LM Studio

# Укажите URL и ключ API для LM Studio
BASE_URL = "http://localhost:1234/v1"  # Предполагается, что сервер LM Studio запущен на этом порту
API_KEY = "meta-llama-3.1-8b-instruct"

# Инициализируем клиента OpenAI для LM Studio
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Функция для загрузки векторного пространства
def load_vector_space(file_path='vector_space.pkl'):
    with open(file_path, 'rb') as f:
        sentences, sentence_embeddings = pickle.load(f)
    return sentences, sentence_embeddings

# Функция для нахождения релевантных предложений
def find_relevant_sections(question, sentences, sentence_embeddings, model):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)
    relevant_sentences = sorted(zip(sentences, scores[0]), key=lambda x: x[1], reverse=True)
    return relevant_sentences

# Функция для создания промпта
def create_prompt(question, vector_space_file='vector_space.pkl'):
    # Загружаем сохранённые вектора и предложения
    sentences, sentence_embeddings = load_vector_space(vector_space_file)
    
    # Загрузка модели для семантического анализа
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Находим релевантные предложения
    relevant_sentences = find_relevant_sections(question, sentences, sentence_embeddings, semantic_model)[:5]
    
    # Объединяем релевантные предложения в контекст
    context = " ".join([sent[0] for sent in relevant_sentences])
    
    # Создаем промпт
    prompt = f"Текст: {context}\n\nВопрос: {question}\nОтвет:"
    return prompt

# Функция для отправки запроса к модели LM Studio
# Функция для отправки запроса к модели LM Studio
def send_prompt_to_model(prompt):
    # Отправка запроса к API LM Studio
    response = client.chat.completions.create(
        model="model-identifier",  # Замените на идентификатор вашей модели
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Обработка ответа
    if response.choices:
        generated_answer = response.choices[0].message.content  # Обновлено здесь
        return generated_answer
    else:
        return "Ошибка: нет ответа от модели."


def main():
    question = "Что самое важное нужно знать про модель памяти?"

    # Создание промпта
    prompt = create_prompt(question, vector_space_file=r'C:\Users\fff\Desktop\NN\vector_space.pkl')
    
    # Отправка промпта к модели LM Studio и получение ответа
    generated_answer = send_prompt_to_model(prompt)
    print(prompt)
    # Выводим сгенерированный ответ
    print(f"Сгенерированный ответ: {generated_answer}")

# Пример использования
if __name__ == "__main__":
    main()
