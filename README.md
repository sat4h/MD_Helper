# AI ассистент для локальной wiki .md

Задача сделать чат-бота, ассистента по локальному .md вики файлу.

Было использовано:

- PyCharm Professional 2024.1.4
- LM Studio
- Telegram (Для бота)

В рамках данной работы [пока что] протестирована одна модель **meta-llama-3.1-8b**

## vectorization.py

С помощью файла [vectorization.py](vectorization.py) собраются все тексты из .md-файлов, разбиваются на предложения и создается для них векторное представление с помощью модели SentenceTransformer. Результаты сохраняются в файл vector_space.pkl

Требуется скачать и импортировать библиотеки

```sh
import os
import nltk
from sentence_transformers import SentenceTransformer
import pickle
```

Здесь указать путь к папке с .md файлами

```sh
if __name__ == "__main__":
    directory = r'C:\Users\fff\Desktop\BD\JavaNotes'  # Путь к вашей папке
    
    # Создание и сохранение векторного пространства
    create_and_save_vector_space(directory, output_file='vector_space.pkl')
```
