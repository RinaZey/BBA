import json
import random
import re
from datetime import datetime
from pathlib import Path
from collections import deque

from modules.tictactoe import TicTacToe
from nlp_utils import clean_text, lemmatize_text, correct_spelling
from intent_classifier import IntentClassifier
from sentiment import get_sentiment
from recommendations import recommend
from dialogue_retrieval import DialogueRetriever


# ──────────── файлы и модели ────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INTENTS_F = DATA_DIR / "intents_dataset.json"
CUSTOM_F = DATA_DIR / "custom_intents.json"
CATALOG_F = DATA_DIR / "product_catalog.json"
DIALOG_F = DATA_DIR / "dialogues.txt"

INTENTS = json.loads(INTENTS_F.read_text("utf-8"))
if CUSTOM_F.exists():
    INTENTS.update(json.loads(CUSTOM_F.read_text("utf-8")))

PRODUCT_CATALOG = json.loads(CATALOG_F.read_text("utf-8"))

# загружаем обученные *.pkl
clf = IntentClassifier(DATA_DIR)
clf.load()
retriever = DialogueRetriever(str(DIALOG_F))

# словарь для spell-checker
DICTIONARY = {
    ex.lower()
    for d in INTENTS.values()
    if isinstance(d, dict)
    for ex in d.get("examples", [])
}
if DIALOG_F.exists():
    for ln in DIALOG_F.read_text("utf-8").splitlines():
        DICTIONARY.update(map(str.lower, re.findall(r"[А-Яа-яA-Za-zё]+", ln)))

# ──────────── реклама / офферы ────────────
AD_COOLDOWN_MSG = 3
AD_COOLDOWN_HOURS = 1

SEASONAL_EVENTS = {
    "11-11": "Чёрная пятница",
    "03-08": "8 марта",
    "23-02": "23 февраля",
}

AD_TRIGGERS = {
    ("сон", "устал", "спал"): (
        "Матрасы",
        None,
        "Кстати, хороший матрас творит чудеса со сном. Хочешь взглянуть?",
    ),
    ("спина", "болит", "поясница"): (
        "Матрасы",
        "ортопедические",
        "Поможет ортопедический матрас с зональной поддержкой 😉",
    ),
    ("переезд", "ремонт", "квартир"): (
        "Кровати",
        None,
        "Новоселье — отличный повод обновить кровать. Подкинуть идеи?",
    ),
}


# ──────────── helpers ────────────
def _save_custom_intents(data: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CUSTOM_F.write_text(json.dumps(data, ensure_ascii=False, indent=4), "utf-8")


def _parse_iso(ts):
    if ts is None or isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


# ──────────── core ────────────
def get_response(text: str, user_data: dict, history: deque) -> str:
    """Главная функция-обработчик диалога."""
    prefs = user_data.setdefault("preferences", {})
    custom_ans = user_data.setdefault("custom_answers", {})
    last_int = user_data.get("last_intent")
    last_bot = user_data.get("last_bot")

    low = text.strip().lower()
    low_clean = re.sub(r"[^а-яёa-z0-9\s]", "", low)

    # --- корректируем типы --------------------------------
    user_data["asked_questions"] = set(user_data.get("asked_questions", []))
    user_data["shown_products"] = set(user_data.get("shown_products", []))
    user_data.setdefault("asked_followup", False)
    user_data.setdefault("msgs_since_ad", 0)
    user_data["last_ad_ts"] = _parse_iso(user_data.get("last_ad_ts"))

    # счётчик сообщений
    user_data["msgs_since_ad"] += 1
    now = datetime.utcnow()
    last_ad_dt = user_data["last_ad_ts"]
    hours_since = ((now - last_ad_dt).total_seconds() / 3600) if last_ad_dt else 1e9

    def _can_offer() -> bool:
        return user_data["msgs_since_ad"] >= AD_COOLDOWN_MSG and hours_since >= AD_COOLDOWN_HOURS

    def _offer(resp: str) -> str:
        user_data.update(last_ad_ts=now.isoformat(), msgs_since_ad=0)
        return resp

    # ------------------------------------------------------
    # small-talk
    if re.search(r"\bкак\s+(дел[аи]|ты)\b", low_clean):
        return random.choice(
            [
                "У меня всё отлично, спасибо! А у тебя как?",
                "Всё хорошо, работаю не покладая транзисторов 😄 А ты?",
            ]
        )
    if "настроени" in low_clean:
        return random.choice(
            [
                "Настроение супер! Как твоё?",
                "Бодрое и весёлое. У тебя какое?",
            ]
        )

    # ожидаемый жанр
    if user_data.get("awaiting_genre"):
        cat = user_data.pop("awaiting_genre")
        reply = recommend(cat, low_clean)
        prefs[f"{cat}_genre"] = low_clean
        user_data.update(last_intent=cat, last_bot=reply)
        return reply

    # сезонные офферы
    mmdd = now.strftime("%m-%d")
    if mmdd in SEASONAL_EVENTS and _can_offer():
        return _offer(
            f"До {SEASONAL_EVENTS[mmdd]} скидка −25 % на матрасы. Показать варианты?"
        )

    # триггерные офферы
    for keys, (cat, sub, pitch) in AD_TRIGGERS.items():
        if any(k in low_clean for k in keys) and _can_offer():
            user_data.update(
                expecting_more_ads=True,
                last_ad_category=cat,
                last_ad_subcategory=sub,
                ad_offer_shown=True,
            )
            if sub:  # сразу показываем товар
                prod = random.choice(PRODUCT_CATALOG[cat][sub])
                return _offer(
                    pitch
                    + f"\n\n*{prod['name']}*\n{prod['description']}\n"
                    f"Цена: {prod['price']} ₽\nПодробнее: {prod['link']}"
                )
            user_data["awaiting_ad_choice"] = True
            return _offer(pitch)

    # сброс режима «ещё»
    if user_data.get("expecting_more_ads") and low not in {"еще", "ещё", "еще раз", "ещё раз"}:
        user_data["expecting_more_ads"] = False
        user_data["shown_products"].clear()

    # пользователь назвал категорию напрямую
    for cat in PRODUCT_CATALOG:
        if re.search(rf"\b{cat.lower()}\b", low_clean):
            user_data.pop("awaiting_ad_choice", None)
            user_data.update(shopping_category=cat, ad_offer_shown=True, ad_offer_done=False)
            subs = ", ".join(PRODUCT_CATALOG[cat])
            return f"Отлично! Какие именно {cat.lower()} интересуют: {subs}?"

    # явный /catalog
    def _catalog_offer() -> str:
        user_data["awaiting_ad_choice"] = True
        return (
            "Кстати, у нас в каталоге есть отличные **кровати** и **матрасы**.\n"
            "Что тебе интереснее: кровати или матрасы?"
        )

    if any(cmd in low for cmd in ("/catalog", "каталог", "товары")) and _can_offer():
        user_data["ad_offer_shown"] = True
        return _offer(_catalog_offer())

    # авто-оффер после 3 реплик
    if (
        len(history) >= 3
        and not user_data.get("ad_offer_shown")
        and not user_data.get("awaiting_ad_choice")
        and _can_offer()
    ):
        user_data["ad_offer_shown"] = True
        return _offer(_catalog_offer())

    # ────────── выбор категории ──────────
    if user_data.get("awaiting_ad_choice"):
        if low_clean in {"нет", "не", "неа", "no"}:
            user_data.pop("awaiting_ad_choice")
            user_data["ad_offer_shown"] = True
            return "Окей! Если захочешь посмотреть каталог — просто скажи 🙂"

        for cat in PRODUCT_CATALOG:
            if low_clean == cat.lower():
                user_data["shopping_category"] = cat
                user_data.pop("awaiting_ad_choice")
                subs = ", ".join(PRODUCT_CATALOG[cat])
                return f"Отлично! Какие именно {cat.lower()} интересуют: {subs}?"

        # неизвестный ответ — снимаем флаг, идём дальше
        user_data.pop("awaiting_ad_choice")

    # подкатегория + 1-я рекомендация
    if "shopping_category" in user_data and "shopping_subcategory" not in user_data:
        cat = user_data["shopping_category"]
        for sub in PRODUCT_CATALOG[cat]:
            if low_clean in {sub.lower(), *sub.lower().split()}:
                user_data.update(
                    last_ad_category=cat,
                    last_ad_subcategory=sub,
                    expecting_more_ads=True,
                    ad_offer_done=True,
                )
                prod = random.choice(PRODUCT_CATALOG[cat][sub])
                user_data["shown_products"].add(prod["name"])
                user_data.pop("shopping_category")
                return (
                    f"Рекомендую: *{prod['name']}*\n\n{prod['description']}\n\n"
                    f"Цена: {prod['price']} ₽\nПодробнее: {prod['link']}"
                )

    # «Ещё» товары
    if user_data.get("expecting_more_ads") and low in {"еще", "ещё", "еще раз", "ещё раз"}:
        cat, sub = user_data["last_ad_category"], user_data["last_ad_subcategory"]
        rest = [p for p in PRODUCT_CATALOG[cat][sub] if p["name"] not in user_data["shown_products"]]
        if not rest:
            user_data["expecting_more_ads"] = False
            user_data["shown_products"].clear()
            return "Пожалуй, это все лучшие варианты 😉"
        prod = random.choice(rest)
        user_data["shown_products"].add(prod["name"])
        return (
            f"Ещё вариант: *{prod['name']}*\n\n{prod['description']}\n\n"
            f"Цена: {prod['price']} ₽\nПодробнее: {prod['link']}"
        )

    # teach-on-the-fly
    if waiting := user_data.get("awaiting_teach"):
        custom_ans[waiting] = text
        user_data.pop("awaiting_teach")
        return random.choice(["Спасибо, запомнил!", "Отлично, принял к сведению!"])

    # пользовательские ответы
    if text in custom_ans:
        return custom_ans[text]

    # любимое X
    if "awaiting_pref_topic" in user_data:
        prefs[user_data.pop("awaiting_pref_topic")] = text
        return f"Спасибо! Я запомнил, что тебе нравится {text}."
    if (m := re.search(r"любим(?:ое|ая|ый|ые)\s+([\w\-а-яё]+)", low_clean)):
        key = f"favorite_{m.group(1)}"
        if key in prefs:
            return f"Мне нравится {prefs[key]}."
        user_data["awaiting_pref_topic"] = key
        return f"А что тебе больше всего нравится в плане {m.group(1)}?"

    # Encore
    if low in {"еще", "ещё", "еще раз", "ещё раз"}:
        if last_int in {"joke", "anecdote", "fun_fact", "fact"}:
            pool = [r for r in INTENTS[last_int]["responses"] if r != last_bot]
            return random.choice(pool) if pool else random.choice(INTENTS[last_int]["responses"])
        if last_int in {"music", "movie", "game", "series"}:
            return recommend(last_int, prefs.get(f"{last_int}_genre"))

    # крестики-нолики
    if "крестики" in low:
        user_data["tic_tac_toe"] = TicTacToe()
        return (
            "Начинаем «крестики-нолики»!\n"
            f"{user_data['tic_tac_toe'].render()}\nТвой ход (A1..C3):"
        )

    # жанр после follow-up
    for cat in {"music", "movie", "game", "series"}:
        if last_int == cat and f"{cat}_genre" not in prefs:
            rec = recommend(cat, text.strip())
            prefs[f"{cat}_genre"] = text.strip()
            user_data["last_bot"] = rec
            return rec

    # sentiment
    lemma = lemmatize_text(
        " ".join(correct_spelling(w, DICTIONARY) for w in clean_text(text).split())
    )
    s = get_sentiment(lemma)
    tone = (
        "Мне очень жаль, что тебе грустно. "
        if s < -0.2
        else "Рад за тебя! "
        if s > 0.5
        else ""
    )

    # intent-predict
    intent = None
    for pred in (lambda x: clf.predict(x), lambda x: clf.predict_fuzzy(x)):
        try:
            cand = pred(lemma)
            if cand in INTENTS:
                intent = cand
                break
        except Exception:
            pass

    # media-intent
    if intent in {"music", "movie", "game", "series"}:
        user_data.update(last_intent=intent, asked_followup=True, awaiting_genre=intent)
        return INTENTS[intent]["follow_up"][0]

    # обычные интенты
    if intent:
        opts = INTENTS[intent]["responses"]
        if last_bot in opts and len(opts) > 1:
            opts = [o for o in opts if o != last_bot]
        resp = random.choice(opts)
        user_data["last_bot"] = resp
        if not user_data["asked_followup"]:
            for f in INTENTS[intent].get("follow_up", []):
                if f not in user_data["asked_questions"]:
                    resp += " " + f
                    user_data["asked_questions"].add(f)
                    user_data["asked_followup"] = True
                    break
        user_data["last_intent"] = intent
        return tone + resp

    # retrieval-ответ
    if candidate := retriever.get_answer(lemma):
        user_data.update(last_bot=candidate, last_intent=None)
        return tone + candidate

    # Teach-fallback
    cid = f"c{re.sub(r'[^a-z0-9]', '', low_clean) or 'intent'}"
    new_i = {
        "examples": [text],
        "responses": ["Я пока не знаю, как на это отвечать. Подскажите пример ответа?"],
    }
    extra = json.loads(CUSTOM_F.read_text("utf-8")) if CUSTOM_F.exists() else {}
    extra[cid] = new_i
    _save_custom_intents(extra)
    INTENTS[cid] = new_i
    user_data["awaiting_teach"] = text
    return "Я пока не знаю, как на это отвечать. Подскажите пример ответа?"
