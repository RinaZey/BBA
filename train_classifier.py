# train_classifier.py
from pathlib import Path
from intent_classifier import IntentClassifier

if __name__ == '__main__':
    data_dir = Path(__file__).parent / 'data'
    clf = IntentClassifier(data_dir)
    clf.train()
    print("Классификатор натренирован и сохранён в intent_vectorizer.pkl, intent_clf.pkl")
