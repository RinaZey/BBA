# sentiment.py
import os
import json
import csv

# Пути к словарям
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
JSON_PATH = os.path.join(DATA_DIR, 'emo_dict.json')
CSV_PATH = os.path.join(DATA_DIR, 'kartaslovsent.csv')

EMO_DICT = {}

# Попытка загрузить JSON
if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH, encoding='utf-8') as f:
            EMO_DICT = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {JSON_PATH}: {e}")
# Иначе — попробовать CSV
elif os.path.exists(CSV_PATH):
    try:
        with open(CSV_PATH, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    word = row[0].strip()
                    try:
                        score = float(row[1].strip())
                    except ValueError:
                        continue
                    EMO_DICT[word] = score
    except Exception as e:
        print(f"Ошибка чтения {CSV_PATH}: {e}")
else:
    print(f"Не найден ни {JSON_PATH}, ни {CSV_PATH}. get_sentiment будет возвращать 0.")

def get_sentiment(text: str) -> float:
    """
    Простейшая тональность: среднее по коэффициентам слов.
    Возвращает значение примерно в диапазоне [-1, +1].
    """
    words = text.split()
    if not words:
        return 0.0
    total = 0.0
    for w in words:
        total += EMO_DICT.get(w, 0.0)
    return total / len(words)
