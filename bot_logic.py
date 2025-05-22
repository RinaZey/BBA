import random
import json
import re
from pathlib import Path
from collections import deque

from modules.tictactoe import TicTacToe
from nlp_utils import clean_text, lemmatize_text, correct_spelling
from intent_classifier import IntentClassifier
from dialogue_retrieval import DialogueRetriever
from sentiment import get_sentiment
from recommendations import recommend  # наш модуль рекомендаций

# ——————————————————————————————————————————————
# 1) Загрузка датасетов и моделей
# ——————————————————————————————————————————————
BASE_DIR            = Path(__file__).parent
DATA_DIR            = BASE_DIR / 'data'
CUSTOM_INTENTS_FILE = DATA_DIR / 'custom_intents.json'

# базовый набор интентов
with open(DATA_DIR / 'intents_dataset.json', encoding='utf-8') as f:
    INTENTS = json.load(f)

# подгружаем пользовательские интенты, если файл есть
if CUSTOM_INTENTS_FILE.exists():
    custom_intents = json.loads(CUSTOM_INTENTS_FILE.read_text("utf-8"))
    INTENTS.update(custom_intents)
else:
    custom_intents = {}

retriever = DialogueRetriever(str(DATA_DIR / 'dialogues.txt'))
clf       = IntentClassifier(DATA_DIR)
clf.load()

# словарь для spell-correction
DICTIONARY = {
    ex.lower()
    for data in INTENTS.values() if isinstance(data, dict)
    for ex in data.get('examples', [])
}

SENT_SPLIT    = re.compile(r'[.?!]+')
QUESTION_WORD = re.compile(r'\b(как|что|где|почему|зачем|когда|куда)\b', re.IGNORECASE)

def _save_custom_intents(data: dict):
    """Сохраняет custom_intents.json внутри папки data."""
    DATA_DIR.mkdir(exist_ok=True)
    CUSTOM_INTENTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

def get_response(text: str, user_data: dict, history: deque) -> str:
    """
    Основная логика + рекомендации + tic-tac-toe +
    teach-fallback + запоминание предпочтений по шаблону «любимое ...».
    """
    # ——— (A) инициализация памяти ———
    prefs        = user_data.setdefault('preferences', {})
    custom_ans   = user_data.setdefault('custom_answers', {})
    last_int     = user_data.get('last_intent')
    asked_fup    = user_data.get('asked_followup', False)
    last_bot     = user_data.get('last_bot')
    waiting      = user_data.get('awaiting_teach', False)
    low          = text.strip().lower()

    # гарантируем, что asked_questions — set
    if 'asked_questions' in user_data and not isinstance(user_data['asked_questions'], set):
        user_data['asked_questions'] = set(user_data['asked_questions'])
    user_data.setdefault('asked_questions', set())

    # ——— (1) Обработка ответа на teach-fallback ———
    if waiting:
        q = user_data.pop('awaiting_teach')
        custom_ans[q] = text
        return random.choice(["Спасибо, запомнил!", "Отлично, принял к сведению!"])

    # ——— (2) Кастом-ответы ———
    if text in custom_ans:
        return custom_ans[text]

    # ——— (3) Универсальный механизм «любимое …» ———
    # 3.1. Если мы спрашивали «любимое X» и ждали ответа:
    if 'awaiting_pref_topic' in user_data:
        topic_key = user_data.pop('awaiting_pref_topic')
        prefs[topic_key] = text
        return f"Спасибо! Я запомнил, что мне нравится {text}."
    # 3.2. Если пользователь спрашивает «любимое X»:
    m = re.search(r'любим(?:ое|ая|ый|ые)\s+([\wа-яё\-]+)', low)
    if m:
        topic = m.group(1)
        key = f"favorite_{topic}"
        # уже знаем?
        if key in prefs:
            return f"Мне нравится {prefs[key]}."
        # не знаем — спрашиваем и запоминаем ключ
        user_data['awaiting_pref_topic'] = key
        return f"Что тебе больше всего нравится в плане {topic}?"

    # ——— (4) Специалька для погоды (оставляем, но теперь уже не критично) ———
    if any(w in low for w in ('погода','солнце','дождь')) and 'weather_preference' in prefs:
        return f"Мне нравится {prefs['weather_preference']}."
    if last_int == 'weather' and any(w in low for w in ('солнце','дождь')):
        choice = 'солнце' if 'солнце' in low else 'дождь'
        prefs['weather_preference'] = choice
        user_data['last_intent']    = None
        user_data['asked_followup'] = False
        return f"Мне нравится {choice}."

    # ——— (5) «еще» для media-рекомендаций ———
    if low in ("еще","ещё","еще раз","ещё раз"):
        if last_int in ("music","movie","game","series"):
            genre = prefs.get(f"{last_int}_genre")
            return recommend(last_int, genre) if genre else "Хочешь ещё рекомендаций? Укажи жанр."
        if last_int in INTENTS:
            return random.choice(INTENTS[last_int]['responses'])

    # ——— (6) запуск крестиков-нолики ———
    if "крестики" in low:
        game = TicTacToe()
        user_data['tic_tac_toe'] = game
        return f"Начинаем «крестики-нолики»!\n{game.render()}\nТвой ход (A1..C3):"

    # ——— (7) рекомендации по жанру после ask-genre ———
    for cat in ("music","movie","game","series"):
        if last_int == cat and f"{cat}_genre" not in prefs:
            genre = text.strip()
            rec   = recommend(cat, genre)
            prefs[f"{cat}_genre"] = genre
            user_data['last_bot'] = rec
            return rec

    # ——— (8) «нет» после follow_up(news) ———
    if low in ('нет','неа','no') and last_int == 'news' and asked_fup:
        user_data['last_intent']    = None
        user_data['asked_followup'] = False
        return "Понятно! О чём хочешь поговорить?"

    # ——— (9) Pre-processing + sentiment → tone ———
    cleaned   = clean_text(text)
    corrected = ' '.join(correct_spelling(w, DICTIONARY) for w in cleaned.split())
    lemma     = lemmatize_text(corrected)
    score     = get_sentiment(lemma)
    tone      = ""
    if score < -0.2:
        tone = "Мне очень жаль, что тебе грустно. "
    elif score > 0.5:
        tone = "Рад за тебя! "

    # ——— (10) intent-predict (прямая + fuzzy) —————————————————
    intent = None
    try:
        c = clf.predict(lemma)
        if c in INTENTS:
            intent = c
    except:
        pass
    if intent is None:
        try:
            c = clf.predict_fuzzy(lemma)
            if c in INTENTS:
                intent = c
        except:
            pass

    # ——— (11) если intent найден — обычный ответ + follow_up —————————————————
    if intent:
        # media-intent → ask-genre
        if intent in ("music","movie","game","series"):
            user_data['last_intent']    = intent
            user_data['asked_followup'] = True
            return INTENTS[intent]['follow_up'][0]

        # pick response
        opts = INTENTS[intent]['responses']
        if last_bot in opts and len(opts) > 1:
            opts = [o for o in opts if o != last_bot]
        resp = random.choice(opts)
        user_data['last_bot'] = resp

        # add follow_up
        if not user_data['asked_followup']:
            for f in INTENTS[intent].get('follow_up', []):
                if f not in user_data['asked_questions']:
                    resp += " " + f
                    user_data['asked_questions'].add(f)
                    user_data['asked_followup'] = True
                    break

        user_data['last_intent'] = intent
        return tone + resp

    # ——— (12) UNIVERSAL teach-fallback (приоритет над retrieval) —————————
    key = re.sub(r'[^a-z0-9]', '', low) or 'intent'
    cid = f"c{key}"
    new_int = {
        "examples":  [text],
        "responses": ["Я пока не знаю, как на это отвечать. Подскажите пример ответа?"]
    }
    data = json.loads(CUSTOM_INTENTS_FILE.read_text("utf-8")) if CUSTOM_INTENTS_FILE.exists() else {}
    data[cid] = new_int
    _save_custom_intents(data)
    INTENTS[cid] = new_int

    user_data['awaiting_teach'] = text
    return "Я пока не знаю, как на это отвечать. Подскажите пример ответа?"
