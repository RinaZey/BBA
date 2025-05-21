# bot_logic.py

import random
import json
import re
from pathlib import Path

from nlp_utils import clean_text, lemmatize_text, correct_spelling
from intent_classifier import IntentClassifier
from dialogue_retrieval import DialogueRetriever
from sentiment import get_sentiment

# ── Загрузка данных ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

with open(DATA_DIR / 'intents_dataset.json', encoding='utf-8') as f:
    INTENTS = json.load(f)

retriever = DialogueRetriever(str(DATA_DIR / 'dialogues.txt'))

clf = IntentClassifier(DATA_DIR)
clf.load()

DICTIONARY = {
    ex.lower()
    for val in INTENTS.values() if isinstance(val, dict)
    for ex in val.get('examples', [])
}

def get_response(text: str):
    """
    Возвращает tuple (response_str, intent_str_or_None).
    """
    # 1) Эмпатия по всему тексту
    full_clean = clean_text(text)
    full_lemma = lemmatize_text(full_clean)
    empathy = ""
    if get_sentiment(full_lemma) < -0.2:
        empathy = "Мне очень жаль, что тебе грустно. "

    # 2) Сплитим ОРИГИНАЛЬНЫЙ текст по [, . ? !]
    parts = [p.strip() for p in re.split(r'[,\.\?!]+', text) if p.strip()]

    seen = set()
    answers = []
    last_intent = None

    for part in parts:
        # 3) Pre-processing
        cleaned = clean_text(part)
        corrected = ' '.join(correct_spelling(w, DICTIONARY) for w in cleaned.split())
        lemma = lemmatize_text(corrected)

        intent = None

        # 4) Прямой SVM-predict
        try:
            cand = clf.predict(lemma)
            if cand in INTENTS:
                intent = cand
        except:
            pass

        # 5) fuzzy-predict
        if not intent:
            try:
                cand = clf.predict_fuzzy(lemma)
                if cand in INTENTS:
                    intent = cand
            except:
                pass

        # 6) fallback dialogues.txt
        if not intent:
            dlg = retriever.get_answer(lemma)
            if dlg:
                answers.append(dlg)
                continue

        # 7) добавляем ответ по интенту (без дублирования)
        if intent and intent not in seen:
            seen.add(intent)
            last_intent = intent
            answers.append(random.choice(INTENTS[intent]['responses']))

    # 8) если совсем ничего — заглушка
    if not answers:
        last_intent = None
        fallback = INTENTS.get(
            'failure_phrases',
            ["Извини, не понял. Попробуй перефразировать."]
        )
        answers.append(random.choice(fallback))

    return empathy + " ".join(answers), last_intent
