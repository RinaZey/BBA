# bot_logic.py

import random
import json
import re
from pathlib import Path

from nlp_utils import clean_text, lemmatize_text, correct_spelling
from intent_classifier import IntentClassifier
from dialogue_retrieval import DialogueRetriever
from sentiment import get_sentiment

# ——————————————————————————————————————————————
# 1) Загрузка данных и моделей
# ——————————————————————————————————————————————
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

with open(DATA_DIR / 'intents_dataset.json', encoding='utf-8') as f:
    INTENTS = json.load(f)

retriever = DialogueRetriever(str(DATA_DIR / 'dialogues.txt'))

clf = IntentClassifier(DATA_DIR)
clf.load()

# Словарь для spell-correction
DICTIONARY = {
    ex.lower()
    for data in INTENTS.values() if isinstance(data, dict)
    for ex in data.get('examples', [])
}

# Регэксп для разделения по предложениям
SENT_SPLIT = re.compile(r'[.?!]+')
# Шаблон «вопросительных» слов
QUESTION_WORD = re.compile(r'\b(как|что|где|почему|зачем|когда|куда)\b', re.IGNORECASE)


def get_response(text: str) -> str:
    """
    1) Препроцессинг всего text
    2) Эмпатия по sentiment
    3) Прямой и fuzzy-predict на всё сообщение
    4) Если интент нашли — сразу ответ
    5) Иначе — ищем «главный вопрос» и отвечаем на него
    6) Фallback по dialogues.txt → заглушка 
    """
    # ——— 1) Pre-processing на весь текст ——
    cleaned_full = clean_text(text)
    corrected_full = ' '.join(correct_spelling(w, DICTIONARY) for w in cleaned_full.split())
    lemma_full = lemmatize_text(corrected_full)

    # ——— 2) Эмпатия ——
    empathy = ""
    if get_sentiment(lemma_full) < -0.2:
        empathy = "Мне очень жаль, что тебе грустно. "

    # ——— 3) Прямой SVM-predict на всё сообщение ——
    intent = None
    try:
        cand = clf.predict(lemma_full)
        if cand in INTENTS and 'responses' in INTENTS[cand]:
            intent = cand
    except:
        intent = None

    # ——— 4) Если прямой не сработал — fuzzy-predict ——
    if not intent:
        try:
            cand = clf.predict_fuzzy(lemma_full)
            if cand in INTENTS and 'responses' in INTENTS[cand]:
                intent = cand
        except:
            intent = None

    # ——— 5) Если нашли интент на весь текст — сразу отвечаем ——
    if intent:
        return empathy + random.choice(INTENTS[intent]['responses'])

    # ——— 6) Иначе разбиваем на предложения и выбираем последний вопрос ——
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]
    # фильтруем только те, что выглядят как вопрос
    questions = [p for p in parts if p.endswith('?') or QUESTION_WORD.search(p)]
    if questions:
        main_q = questions[-1]
    else:
        # нет явного вопроса — уточняем
        return empathy + random.choice([
            "Прости, не совсем понял, можешь перефразировать вопрос?",
            "Какой именно вопрос тебя интересует?",
            "Можешь сформулировать вопрос чуть точнее?"
        ])

    # ——— 7) Предобработка и классификация одного вопроса ——
    cleaned = clean_text(main_q)
    corrected = ' '.join(correct_spelling(w, DICTIONARY) for w in cleaned.split())
    lemma = lemmatize_text(corrected)

    # прямой для вопроса
    try:
        cand = clf.predict(lemma)
        if cand in INTENTS and 'responses' in INTENTS[cand]:
            return empathy + random.choice(INTENTS[cand]['responses'])
    except:
        pass

    # fuzzy для вопроса
    try:
        cand = clf.predict_fuzzy(lemma)
        if cand in INTENTS and 'responses' in INTENTS[cand]:
            return empathy + random.choice(INTENTS[cand]['responses'])
    except:
        pass

    # ——— 8) dialogues.txt fallback ——
    reply = retriever.get_answer(lemma)
    if reply:
        return empathy + reply

    # ——— 9) Финальная заглушка ——
    fallback = INTENTS.get('failure_phrases', ["Извини, не понял. Попробуй перефразировать."])
    return empathy + random.choice(fallback)
