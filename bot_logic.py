# bot_logic.py

import random
import json
import re
from pathlib import Path
from collections import deque

from nlp_utils import clean_text, lemmatize_text, correct_spelling
from intent_classifier import IntentClassifier
from dialogue_retrieval import DialogueRetriever
from sentiment import get_sentiment

from recommendations import recommend  # модуль рекомендаций из примера выше

# ——————————————————————————————————————————————
# 1) Загрузка датасетов и моделей
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

# Разбиение на предложения и шаблон вопроса
SENT_SPLIT    = re.compile(r'[.?!]+')
QUESTION_WORD = re.compile(r'\b(как|что|где|почему|зачем|когда|куда)\b', re.IGNORECASE)

def get_response(text: str, user_data: dict) -> str:
    """
    1) On-the-fly обучение
    2) Смена ника ("Зови меня ...")
    3) Рекомендации (фильмы / музыка / игры / сериалы)
    4) Preprocess + sentiment → тон
    5) Intent → ответ + разовый follow_up
    6) Последний вопрос → ответ
    7) Диалоговый fallback → teach
    """
    # A) Контекст
    history  = user_data.setdefault('history', deque(maxlen=10))
    custom   = user_data.setdefault('custom_answers', {})

    # B) On-the-fly обучение
    if 'awaiting_teach' in user_data:
        q = user_data.pop('awaiting_teach')
        custom[q] = text
        return random.choice(["Спасибо, запомнил!", "Отлично, принял к сведению!"])

    history.append(text)

    # C) "Зови меня ..." / "Меня зовут ..."
    m = re.match(r'^(?:зови меня|меня зовут)\s+["«]?(.+?)["»]?$',
                 text.strip(), flags=re.IGNORECASE)
    if m:
        nickname = m.group(1)
        user_data['username'] = nickname
        return f"Приятно познакомиться, {nickname}! Чем любишь заниматься в свободное время?"

    # D) Кастом-ответы
    if text in custom:
        return custom[text]

    # E) Рекомендации по жанру
    low = text.lower()
    if 'awaiting_genre' in user_data:
        cat = user_data.pop('awaiting_genre')
        genre = text.strip()
        return recommend(cat, genre)

    # музыка
    m = re.search(r'порекомендуй(?:те)?\s+(\w+)\s+музык', low)
    if m:
        return recommend("music", m.group(1))
    if re.search(r'порекомендуй(?:те)?\s+музык', low):
        user_data['awaiting_genre'] = "music"
        return "Конечно! Какой жанр музыки тебе интересен?"

    # фильмы
    m = re.search(r'порекомендуй(?:те)?\s+(\w+)\s*(?:фильм|кино)', low)
    if m:
        return recommend("movie", m.group(1))
    if re.search(r'порекомендуй(?:те)?\s*(?:фильм|кино)', low):
        user_data['awaiting_genre'] = "movie"
        return "С удовольствием! Какой жанр фильмов?"

    # сериалы
    m = re.search(r'порекомендуй(?:те)?\s+(\w+)\s*сериал', low)
    if m:
        return recommend("series", m.group(1))
    if re.search(r'порекомендуй(?:те)?\s*сериал', low):
        user_data['awaiting_genre'] = "series"
        return "Конечно! Какой жанр сериалов тебе больше по душе?"

    # игры
    m = re.search(r'порекомендуй(?:те)?\s+(\w+)\s*игр', low)
    if m:
        return recommend("game", m.group(1))
    if re.search(r'порекомендуй(?:те)?\s*игр', low):
        user_data['awaiting_genre'] = "game"
        return "Понял! А какой жанр игр тебе интересен?"

    # F) Preprocess + sentiment
    cleaned   = clean_text(text)
    corrected = ' '.join(correct_spelling(w, DICTIONARY) for w in cleaned.split())
    lemma     = lemmatize_text(corrected)

    score = get_sentiment(lemma)
    if score < -0.2:
        tone = "Мне очень жаль, что тебе грустно. "
    elif score > 0.5:
        tone = "Здорово слышать! "
    else:
        tone = ""

    # G0) Точное совпадение с примерами
    normalized = corrected.lower().strip()
    intent = None
    for key, data in INTENTS.items():
        if any(clean_text(ex).lower() == normalized for ex in data.get('examples', [])):
            intent = key
            break

    # G1) При грусти — форсируем "depression"
    if intent is None and score < -0.2 and 'depression' in INTENTS:
        intent = 'depression'

    # G2) SVM / fuzzy
    if intent is None:
        try:
            cand = clf.predict(lemma)
            if cand in INTENTS:
                intent = cand
        except:
            pass
    if intent is None:
        try:
            cand = clf.predict_fuzzy(lemma)
            if cand in INTENTS:
                intent = cand
        except:
            pass

    # H) Функция один-разовый follow_up
    def apply_follow_up(resp_text: str, key: str) -> str:
        last  = user_data.get('last_intent')
        asked = user_data.get('asked_followup', False)
        if key != last:
            user_data['asked_followup'] = False
            asked = False
        if not asked:
            fups = INTENTS[key].get('follow_up', [])
            if fups:
                # подставим имя пользователя, если есть
                nick = user_data.get('username')
                extra = random.choice(fups)
                if nick:
                    extra = extra.replace("{username}", nick)
                resp_text += " " + extra
                user_data['asked_followup'] = True
        user_data['last_intent'] = key
        return resp_text

    # I) Ответ по intent
    if intent:
        resp = random.choice(INTENTS[intent]['responses'])
        resp = apply_follow_up(resp, intent)
        return tone + resp

    # J) Последний вопрос → ответ
    parts     = [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]
    questions = [p for p in parts if p.endswith('?') or QUESTION_WORD.search(p)]
    if questions:
        main_q = questions[-1]
        cln    = clean_text(main_q)
        corr   = ' '.join(correct_spelling(w, DICTIONARY) for w in cln.split())
        lemq   = lemmatize_text(corr)

        # direct
        try:
            cand = clf.predict(lemq)
            if cand in INTENTS:
                resp = random.choice(INTENTS[cand]['responses'])
                resp = apply_follow_up(resp, cand)
                return tone + resp
        except:
            pass
        # fuzzy
        try:
            cand = clf.predict_fuzzy(lemq)
            if cand in INTENTS:
                resp = random.choice(INTENTS[cand]['responses'])
                resp = apply_follow_up(resp, cand)
                return tone + resp
        except:
            pass

    # K) Диалоговый fallback
    reply = retriever.get_answer(lemma)
    if reply:
        return tone + reply

    # L) Teach fallback
    user_data['awaiting_teach'] = text
    return tone + random.choice(
        INTENTS.get('failure_phrases',
                    ["Извини, не понял. Попробуй перефразировать."])
    )
