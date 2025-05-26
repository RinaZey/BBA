import random, json, re
from pathlib            import Path
from collections        import deque

from modules.tictactoe  import TicTacToe
from nlp_utils          import clean_text, lemmatize_text, correct_spelling
from intent_classifier  import IntentClassifier
from sentiment          import get_sentiment
from recommendations    import recommend          # ваш модуль с рекомендациями
# ──────────────────────────────────────────────────────────────
# 1) данные и модели
# ──────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).parent
DATA_DIR            = BASE_DIR / 'data'
CUSTOM_INTENTS_FILE = DATA_DIR / 'custom_intents.json'
CATALOG_FILE        = DATA_DIR / 'product_catalog.json'

with open(DATA_DIR / 'intents_dataset.json', encoding='utf-8') as f:
    INTENTS = json.load(f)
if CUSTOM_INTENTS_FILE.exists():
    INTENTS.update(json.loads(CUSTOM_INTENTS_FILE.read_text('utf-8')))

with open(CATALOG_FILE, encoding='utf-8') as f:
    PRODUCT_CATALOG = json.load(f)

clf = IntentClassifier(DATA_DIR)
clf.load()

DICTIONARY = {
    ex.lower()
    for data in INTENTS.values() if isinstance(data, dict)
    for ex in data.get('examples', [])
}

# ──────────────────────────────────────────────────────────────
def _save_custom_intents(data: dict):
    DATA_DIR.mkdir(exist_ok=True)
    CUSTOM_INTENTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding='utf-8'
    )

# ──────────────────────────────────────────────────────────────
def get_response(text: str, user_data: dict, history: deque) -> str:
    """
    Главная логика ответа бота.
    """
    # —── краткие ссылки на память пользователя —──
    prefs        = user_data.setdefault('preferences', {})
    custom_ans   = user_data.setdefault('custom_answers', {})
    last_int     = user_data.get('last_intent')
    asked_fup    = user_data.get('asked_followup', False)
    last_bot     = user_data.get('last_bot')
    waiting_teach = user_data.get('awaiting_teach', False)

    low       = text.strip().lower()
    low_clean = re.sub(r'[^а-яёa-z0-9\s]', '', low)

    # ——— корректируем типы множеств в user_data ———
    if not isinstance(user_data.get('asked_questions'), set):
        user_data['asked_questions'] = set(user_data.get('asked_questions', []))

    if not isinstance(user_data.get('shown_products'), set):
        user_data['shown_products'] = set(user_data.get('shown_products', []))

    AFFIRM = {'да', 'ага', 'ок', 'окей', 'конечно', 'хорошо', 'давай', 'хочу'}

    # ────────── 0. small-talk «как дела» / «как настроение» ──────────
    if re.search(r'\bкак\s+(дел[аи]|ты)\b', low_clean):
        return random.choice([
            "У меня всё отлично, спасибо! А у тебя как?",
            "Всё хорошо, работаю не покладая транзисторов 😄 А ты?"
        ])
    if 'настроени' in low_clean:
        return random.choice([
            "Настроение супер! Как твоё?",
            "Бодрое и весёлое. У тебя какое?"
        ])

    # ────────── 0a. Обработка ожидаемого ЖАНРА (movie/music/…) ──────────
    if user_data.get('awaiting_genre'):
        cat   = user_data.pop('awaiting_genre')       # movie | music | game | series
        genre = low_clean
        reply = recommend(cat, genre)

        prefs[f"{cat}_genre"] = genre                 # запоминаем жанр
        user_data['last_intent'] = cat
        user_data['last_bot']    = reply
        return reply

    # ────────── 0b. сброс режима «ещё» для рекламы ──────────
    if user_data.get('expecting_more_ads') and low not in {"еще", "ещё", "еще раз", "ещё раз"}:
        user_data['expecting_more_ads'] = False
        user_data['shown_products'].clear()

        # ────────── 0b. пользователь уже назвал категорию (кровати/матрасы) ──────────
    for cat in PRODUCT_CATALOG:                               # cat = "Кровати", "Матрасы"
        # ищем слово-категорию внутри текста по границе слова
        if re.search(rf'\b{cat.lower()}\b', low_clean):
            # если до этого бот спрашивал «кровати или матрасы» – сбрасываем ожидание
            user_data.pop('awaiting_ad_choice', None)

            user_data.update(
                shopping_category = cat,      # запоминаем выбор
                ad_offer_done     = False,    # начинаем новый подбор
                ad_offer_shown    = True      # помечаем, что реклама уже активна
            )
            subcats = ", ".join(PRODUCT_CATALOG[cat].keys())
            return f"Отлично! Какие именно {cat.lower()} тебе интересны: {subcats}?"

    # ────────── 1. запуск рекламы ──────────
    def advert_prompt() -> str:
        
        user_data['awaiting_ad_choice'] = True
        return ("Кстати, у нас в каталоге есть отличные **кровати** и **матрасы**.\n"
                "Что тебе интереснее: кровати или матрасы?")

    if (low_clean in AFFIRM and 'подберу кровать' in (last_bot or '').lower()):
        user_data['ad_offer_shown'] = True
        return advert_prompt()

    if any(k in low for k in ('/catalog', 'каталог', 'товары')):
        user_data['ad_offer_shown'] = True
        return advert_prompt()

    if len(history) >= 3 and not user_data.get('ad_offer_shown') \
       and not user_data.get('awaiting_ad_choice'):
        user_data['ad_offer_shown'] = True
        return advert_prompt()

    # ────────── 2. реклама: выбор категории ──────────
    if user_data.get('awaiting_ad_choice'):
        if low_clean in AFFIRM:
            return "Выбери, пожалуйста: кровати или матрасы?"
        for cat in PRODUCT_CATALOG:
            if low_clean == cat.lower():
                user_data['shopping_category'] = cat
                del user_data['awaiting_ad_choice']
                subs = ", ".join(PRODUCT_CATALOG[cat].keys())
                return f"Отлично! Какие именно {cat.lower()} интересуют: {subs}?"
        del user_data['awaiting_ad_choice']   # ответ был непонятен

    # пользователь спонтанно ввёл «кровати» / «матрасы»
    if low_clean in (c.lower() for c in PRODUCT_CATALOG):
        user_data.update(
            shopping_category = next(c for c in PRODUCT_CATALOG if c.lower() == low_clean),
            ad_offer_done     = False
        )
        cat  = user_data['shopping_category']
        subs = ", ".join(PRODUCT_CATALOG[cat].keys())
        return f"Хорошо! Какие {cat.lower()} тебе интересны: {subs}?"

    # ────────── 3. реклама: подкатегория + первая рекомендация ──────────
    if 'shopping_category' in user_data and 'shopping_subcategory' not in user_data:
        cat = user_data['shopping_category']
        for sub in PRODUCT_CATALOG[cat]:
            if low_clean == sub.lower() or sub.lower() in low_clean or low_clean in sub.lower():
                user_data.update(
                    last_ad_category    = cat,
                    last_ad_subcategory = sub,
                    expecting_more_ads  = True,
                    ad_offer_done       = True
                )
                prod = random.choice(PRODUCT_CATALOG[cat][sub])
                user_data['shown_products'].add(prod['name'])
                del user_data['shopping_category']
                return (f"Рекомендую: *{prod['name']}*\n{prod['description']}\n"
                        f"Цена: {prod['price']} руб.\nПодробнее: {prod['link']}")

    # ────────── 3b. реклама: «Ещё» ──────────
    if user_data.get('expecting_more_ads') and low in {"еще", "ещё", "еще раз", "ещё раз"}:
        cat   = user_data.get('last_ad_category')
        sub   = user_data.get('last_ad_subcategory')
        shown = user_data['shown_products']
        if cat and sub:
            rest = [p for p in PRODUCT_CATALOG[cat][sub] if p['name'] not in shown]
            if not rest:
                user_data['expecting_more_ads'] = False
                shown.clear()
                return "Пожалуй, это все лучшие варианты в этой категории ;)"
            prod = random.choice(rest)
            shown.add(prod['name'])
            return (f"Ещё вариант: *{prod['name']}*\n{prod['description']}\n"
                    f"Цена: {prod['price']} руб.\nПодробнее: {prod['link']}")

    # ────────── 4. teach-on-the-fly (ответ пользователя) ──────────
    if waiting_teach:
        q = user_data.pop('awaiting_teach')
        custom_ans[q] = text
        return random.choice(["Спасибо, запомнил!", "Отлично, принял к сведению!"])

    # ────────── 5. ответы из пользовательских заготовок ──────────
    if text in custom_ans:
        return custom_ans[text]

    # ────────── 6. «любимое X» ──────────
    if 'awaiting_pref_topic' in user_data:
        key = user_data.pop('awaiting_pref_topic')
        prefs[key] = text
        return f"Спасибо! Я запомнил, что мне нравится {text}."

    m = re.search(r'любим(?:ое|ая|ый|ые)\s+([\w\-а-яё]+)', low_clean)
    if m:
        topic = m.group(1)
        key   = f"favorite_{topic}"
        if key in prefs:
            return f"Мне нравится {prefs[key]}."
        user_data['awaiting_pref_topic'] = key
        return f"А что тебе больше всего нравится в плане {topic}?"

    # ────────── 7. погода ──────────
    if any(w in low for w in ('погода', 'солнце', 'дождь')) and 'weather_preference' in prefs:
        return f"Мне нравится {prefs['weather_preference']}."
    if last_int == 'weather' and any(w in low for w in ('солнце', 'дождь')):
        choice = 'солнце' if 'солнце' in low else 'дождь'
        prefs['weather_preference'] = choice
        user_data['last_intent'] = None
        user_data['asked_followup'] = False
        return f"Мне нравится {choice}."

    # ────────── 8. Encore («ещё») для шуток / фактов / медиа ──────────
    REPEATABLE = {'joke', 'jokes', 'anecdote', 'fun_fact', 'fact'}
    if low in {"еще", "ещё", "еще раз", "ещё раз"}:
        if last_int in REPEATABLE:
            pool = [r for r in INTENTS[last_int]['responses'] if r != last_bot]
            return random.choice(pool) if pool else random.choice(INTENTS[last_int]['responses'])
        if last_int in {"music", "movie", "game", "series"}:
            genre = prefs.get(f"{last_int}_genre")
            return recommend(last_int, genre) if genre else "Хочешь ещё рекомендаций? Укажи жанр."

    # ────────── 9. крестики-нолики ──────────
    if "крестики" in low:
        game = TicTacToe()
        user_data['tic_tac_toe'] = game
        return f"Начинаем «крестики-нолики»!\n{game.render()}\nТвой ход (Формат СТРОГО A1..C3):"

    # ────────── 10. жанр после follow-up (media) ──────────
    for cat in {"music", "movie", "game", "series"}:
        if last_int == cat and f"{cat}_genre" not in prefs:
            genre = text.strip()
            rec   = recommend(cat, genre)
            prefs[f"{cat}_genre"] = genre
            user_data['last_bot'] = rec
            return rec

    # ────────── 11. «нет» после follow-up (news) ──────────
    if low in {'нет', 'неа', 'no'} and last_int == 'news' and asked_fup:
        user_data['last_intent'] = None
        user_data['asked_followup'] = False
        return "Понятно! О чём хочешь поговорить?"

    # ────────── 12. sentiment-тональность ──────────
    cleaned   = clean_text(text)
    corrected = ' '.join(correct_spelling(w, DICTIONARY) for w in cleaned.split())
    lemma     = lemmatize_text(corrected)
    score     = get_sentiment(lemma)
    tone      = "Мне очень жаль, что тебе грустно. " if score < -0.2 else \
                "Рад за тебя! "                       if score >  0.5 else ""

    # ────────── 13. intent-предсказание ──────────
    intent = None
    try:
        cand = clf.predict(lemma)
        if cand in INTENTS:
            intent = cand
    except Exception:
        pass
    if intent is None:
        try:
            cand = clf.predict_fuzzy(lemma)
            if cand in INTENTS:
                intent = cand
        except Exception:
            pass

    # ────────── 13a. MEDIA-интенты ──────────
    if intent in {"music", "movie", "game", "series"}:
        user_data['last_intent']    = intent
        user_data['asked_followup'] = True
        user_data['awaiting_genre'] = intent
        return INTENTS[intent]['follow_up'][0]

    # ────────── 13b. обычные интенты ──────────
    if intent:
        opts = INTENTS[intent]['responses']
        if last_bot in opts and len(opts) > 1:
            opts = [o for o in opts if o != last_bot]
        resp = random.choice(opts)
        user_data['last_bot'] = resp

        if not user_data['asked_followup']:
            for f in INTENTS[intent].get('follow_up', []):
                if f not in user_data['asked_questions']:
                    resp += " " + f
                    user_data['asked_questions'].add(f)
                    user_data['asked_followup'] = True
                    break

        user_data['last_intent'] = intent
        return tone + resp

    # ────────── 14. Teach-fallback ──────────
    key = re.sub(r'[^a-z0-9]', '', low_clean) or 'intent'
    cid = f"c{key}"
    new_i = {
        "examples":  [text],
        "responses": ["Я пока не знаю, как на это отвечать. Подскажите, пример ответа?"]
    }
    data = {}
    if CUSTOM_INTENTS_FILE.exists():
        data = json.loads(CUSTOM_INTENTS_FILE.read_text('utf-8'))
    data[cid] = new_i
    _save_custom_intents(data)
    INTENTS[cid] = new_i

    user_data['awaiting_teach'] = text
    return "Я пока не знаю, как на это отвечать. Подскажите, пример ответа?"
