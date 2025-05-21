# intent_classifier.py

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.metrics.distance import edit_distance

from nlp_utils import clean_text, lemmatize_text

class IntentClassifier:
    def __init__(self, data_dir: Path):
        # 1) Загружаем «сырые» интенты
        with open(data_dir / 'intents_dataset.json', encoding='utf-8') as f:
            raw = json.load(f)

        # Оставляем только словари с key 'examples'
        self.intents = {
            name: val for name, val in raw.items()
            if isinstance(val, dict) and 'examples' in val
        }

        # 2) Нормализуем (clean+lemmatize) все примеры сразу
        self.norm_examples = {
            intent: [lemmatize_text(clean_text(ex)) for ex in data['examples']]
            for intent, data in self.intents.items()
        }

        # 3) Векторайзер и SVM
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,3))
        self.clf = LinearSVC()

    def train(self):
        # Готовим X и y из нормированных примеров
        X, y = [], []
        for intent, examples in self.norm_examples.items():
            for ex in examples:
                X.append(ex)
                y.append(intent)

        # Обучаем
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)

        # Сохраняем на диск
        with open('intent_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('intent_clf.pkl', 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self):
        # Загружаем модель и векторайзер
        with open('intent_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open('intent_clf.pkl', 'rb') as f:
            self.clf = pickle.load(f)

    def predict(self, text: str) -> str:
        # Прямой predict: чистим+лемматизируем и кидаем в SVM
        norm = lemmatize_text(clean_text(text))
        return self.clf.predict(self.vectorizer.transform([norm]))[0]

    def predict_fuzzy(self, text: str, threshold: float = 0.25) -> str:
        # Фаззи-fallback: смотрим уверенность SVM, иначе ищем ближайший пример
        norm = lemmatize_text(clean_text(text))

        # 1) Проверяем расстояние до примеров предсказанного интента
        intent = self.predict(text)
        exs = self.norm_examples[intent]
        dist0 = min(edit_distance(norm, ex) / max(1, len(ex)) for ex in exs)
        if dist0 < threshold:
            return intent

        # 2) Ищем самый близкий пример вообще
        best_intent, best_d = None, threshold
        for cand_intent, examples in self.norm_examples.items():
            for ex in examples:
                d = edit_distance(norm, ex) / max(1, len(ex))
                if d < best_d:
                    best_d = d
                    best_intent = cand_intent

        return best_intent
