# bot_logic.py
# ---------------------------------------------------------------------------
#  ✔️  Текст или голос  → STT (Vosk) → get_response (NLP-логика) → TTS-WAV
#     → конверсия в OGG/Opus  → voice-сообщение в Telegram.
# ---------------------------------------------------------------------------

import os, json, random, re, pyttsx3
from datetime import datetime
from pathlib   import Path
from collections import deque

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Updater, CommandHandler, MessageHandler, Filters, CallbackContext,
)
from pydub import AudioSegment

# ────────── внутренние модули ──────────────────────────────────────────────
from modules.tictactoe import TicTacToe
from nlp_utils          import clean_text, lemmatize_text, correct_spelling
from intent_classifier   import IntentClassifier
from sentiment           import get_sentiment
from recommendations     import recommend
from dialogue_retrieval  import DialogueRetriever
from audio_utils         import stt_from_wav                 # офлайн-Vosk

# ────────── окружение / каталоги ───────────────────────────────────────────
load_dotenv()
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"; TEMP_DIR.mkdir(exist_ok=True)

INTENTS_F = DATA_DIR / "intents_dataset.json"
CUSTOM_F  = DATA_DIR / "custom_intents.json"
CATALOG_F = DATA_DIR / "product_catalog.json"
DIALOG_F  = DATA_DIR / "dialogues.txt"

# ────────── данные и ML-модели ─────────────────────────────────────────────
INTENTS = json.loads(INTENTS_F.read_text('utf-8'))
if CUSTOM_F.exists():
    INTENTS.update(json.loads(CUSTOM_F.read_text('utf-8')))
PRODUCT_CATALOG = json.loads(CATALOG_F.read_text('utf-8'))

clf       = IntentClassifier(DATA_DIR);  clf.load()
retriever = DialogueRetriever(str(DIALOG_F))

DICTIONARY = {ex.lower() for d in INTENTS.values() if isinstance(d, dict)
                             for ex in d.get("examples", [])}
if DIALOG_F.exists():
    for ln in DIALOG_F.read_text('utf-8').splitlines():
        DICTIONARY.update(re.findall(r"[а-яёa-z]+", ln.lower()))

# ────────── TTS (pyttsx3 → WAV) ────────────────────────────────────────────
tts = pyttsx3.init()
for v in tts.getProperty("voices"):
    if "russian" in v.name.lower() and "male" in v.name.lower():
        tts.setProperty("voice", v.id); break
tts.setProperty("rate", 140)

def _tts_to_wav(text: str, path: Path):
    tts.save_to_file(text, str(path))
    tts.runAndWait()

# ────────── утилиты ────────────────────────────────────────────────────────
def _parse_iso(ts):
    try: return datetime.fromisoformat(ts) if isinstance(ts, str) else ts
    except: return None

def _save_custom_intents(data: dict):
    CUSTOM_F.write_text(json.dumps(data, ensure_ascii=False, indent=4), 'utf-8')

# ────────── параметры рекламы ──────────────────────────────────────────────
AD_COOLDOWN_MSG, AD_COOLDOWN_HOURS = 3, 1
SEASONAL_EVENTS = {"11-11":"Чёрная пятница","03-08":"8 марта","23-02":"23 февраля"}
AD_TRIGGERS = {
    ("сон","устал","спал"): ("Матрасы",None,"Кстати, хороший матрас творит чудеса со сном. Хочешь взглянуть?"),
    ("спина","болит","поясница"): ("Матрасы",None,"Поможет ортопедический матрас с зональной поддержкой 😉"),
    ("переезд","ремонт","квартир"): ("Кровати",None,"Новоселье — отличный повод обновить кровать. Подкинуть идеи?"),
}

# ────────── ГЛАВНАЯ логика ответа ──────────────────────────────────────────
def get_response(text: str, user_data: dict, history: deque) -> str:
    prefs      = user_data.setdefault("preferences", {})
    custom_ans = user_data.setdefault("custom_answers", {})
    last_int   = user_data.get("last_intent")
    last_bot   = user_data.get("last_bot")

    low        = text.lower().strip()
    low_clean  = re.sub(r"[^а-яёa-z0-9\s]", "", low)

    # счётчики / типы --------------------------------------------------------
    user_data["asked_questions"] = set(user_data.get("asked_questions", []))
    user_data["shown_products"]  = set(user_data.get("shown_products",  []))
    user_data.setdefault("asked_followup", False)
    user_data.setdefault("msgs_since_ad",   0)
    user_data["last_ad_ts"] = _parse_iso(user_data.get("last_ad_ts"))

    user_data["msgs_since_ad"] += 1
    now = datetime.utcnow()
    hours_since = ((now - user_data["last_ad_ts"]).total_seconds()/3600) if user_data["last_ad_ts"] else 1e9

    def _can_offer(): return user_data["msgs_since_ad"]>=AD_COOLDOWN_MSG and hours_since>=AD_COOLDOWN_HOURS
    def _offer(r): user_data.update(last_ad_ts=now.isoformat(), msgs_since_ad=0); return r

    # 0. hello/bye через ML-классификатор
    try:
        i0 = clf.predict_intent(text)
        if i0 in INTENTS and i0 in {"hello","bye"}:
            r = random.choice(INTENTS[i0]["responses"])
            user_data.update(last_intent=i0,last_bot=r)
            return r
    except: pass

    # 1. small-talk
    if re.search(r"\bкак\s+(дел[аи]|ты)\b", low_clean):
        return random.choice(["У меня всё отлично, спасибо! А у тебя?","Всё хорошо, работаю не покладая транзисторов 😄 А ты?"])
    if "настроени" in low_clean:
        return random.choice(["Настроение супер! Как твоё?","Бодрое и весёлое. У тебя какое?"])

    # 2. ожидаемый жанр
    if user_data.get("awaiting_genre"):
        cat   = user_data.pop("awaiting_genre")
        reply = recommend(cat, low_clean)
        prefs[f"{cat}_genre"] = low_clean
        user_data.update(last_intent=cat,last_bot=reply)
        return reply

    # 3. сезонные офферы
    if now.strftime("%m-%d") in SEASONAL_EVENTS and _can_offer():
        return _offer(f"До {SEASONAL_EVENTS[now.strftime('%m-%d')]} скидка –25 % на матрасы. Показать варианты?")

    # 4. триггерные офферы
    for keys,(cat,sub,pitch) in AD_TRIGGERS.items():
        if any(k in low_clean for k in keys) and _can_offer():
            user_data.update(expect_more=True,last_ad_category=cat,last_ad_subcategory=sub,ad_offer_shown=True)
            if sub:
                prod=random.choice(PRODUCT_CATALOG[cat][sub])
                return _offer(pitch+f"\n\n*{prod['name']}*\n{prod['description']}\nЦена: {prod['price']} ₽\nПодробнее: {prod['link']}")
            user_data["awaiting_ad_choice"]=True
            return _offer(pitch)

    # 5. сброс режима «ещё»
    if user_data.get("expect_more") and low not in {"еще","ещё","еще раз","ещё раз"}:
        user_data["expect_more"]=False; user_data["shown_products"].clear()

    # 6. прямое упоминание категории
    for cat in PRODUCT_CATALOG:
        if re.search(rf"\b{cat.lower()}\b", low_clean):
            user_data.pop("awaiting_ad_choice",None)
            user_data.update(shop_cat=cat, ad_offer_shown=True)
            return f"Отлично! Какие именно {cat.lower()} интересуют: {', '.join(PRODUCT_CATALOG[cat])}?"

    # 7. явная команда каталога
    def _catalog_offer():
        user_data["awaiting_ad_choice"]=True
        return "В каталоге есть отличные **кровати** и **матрасы**. Что интереснее?"
    if any(cmd in low for cmd in ("/catalog","каталог","товары")) and _can_offer():
        user_data["ad_offer_shown"]=True
        return _offer(_catalog_offer())

    # 8. авто-оффер
    if len(history)>=3 and not user_data.get("ad_offer_shown") and not user_data.get("awaiting_ad_choice") and _can_offer():
        user_data["ad_offer_shown"]=True
        return _offer(_catalog_offer())

    # 9. выбор категории
    if user_data.get("awaiting_ad_choice"):
        if low_clean in {"нет","не","неа","no"}:
            user_data.pop("awaiting_ad_choice"); user_data["ad_offer_shown"]=True
            return "Хорошо! Скажи, когда захочешь посмотреть каталог 🙂"
        for cat in PRODUCT_CATALOG:
            if low_clean==cat.lower():
                user_data.update(shop_cat=cat); user_data.pop("awaiting_ad_choice")
                return f"Какие именно {cat.lower()} интересуют: {', '.join(PRODUCT_CATALOG[cat])}?"
        user_data.pop("awaiting_ad_choice")

    # 10. первая рекомендация
    if "shop_cat" in user_data and "shop_sub" not in user_data:
        cat=user_data["shop_cat"]
        for sub in PRODUCT_CATALOG[cat]:
            if low_clean in {sub.lower(),*sub.lower().split()}:
                user_data.update(last_ad_category=cat,last_ad_subcategory=sub,expect_more=True)
                prod=random.choice(PRODUCT_CATALOG[cat][sub])
                user_data["shown_products"].add(prod["name"])
                user_data.pop("shop_cat")
                return f"Рекомендую: *{prod['name']}*\n\n{prod['description']}\n\nЦена: {prod['price']} ₽\nПодробнее: {prod['link']}"

    # 11. «ещё»
    if user_data.get("expect_more") and low in {"еще","ещё","еще раз","ещё раз"}:
        cat,sub=user_data["last_ad_category"], user_data["last_ad_subcategory"]
        rest=[p for p in PRODUCT_CATALOG[cat][sub] if p["name"] not in user_data["shown_products"]]
        if not rest:
            user_data["expect_more"]=False; user_data["shown_products"].clear()
            return "Пожалуй, это все лучшие варианты 😉"
        prod=random.choice(rest); user_data["shown_products"].add(prod["name"])
        return f"Ещё вариант: *{prod['name']}*\n\n{prod['description']}\n\nЦена: {prod['price']} ₽\nПодробнее: {prod['link']}"

    # 12. интерактивное обучение
    if waiting:=user_data.get("awaiting_teach"):
        custom_ans[waiting]=text; user_data.pop("awaiting_teach")
        return random.choice(["Спасибо, запомнил!","Отлично, принял к сведению!"])
    if text in custom_ans:
        return custom_ans[text]

    # 13. «Любимое X»
    if "awaiting_pref_topic" in user_data:
        prefs[user_data.pop("awaiting_pref_topic")]=text
        return f"Спасибо! Запомнил, что тебе нравится {text}."
    if (m:=re.search(r"любим(?:ое|ая|ый|ые)\s+([\w\-а-яё]+)", low_clean)):
        k=f"favorite_{m.group(1)}"
        if k in prefs: return f"Мне нравится {prefs[k]}."
        user_data["awaiting_pref_topic"]=k
        return f"А что тебе больше всего нравится в плане {m.group(1)}?"

    # 14. encore («ещё»)
    if low in {"еще","ещё","еще раз","ещё раз"}:
        if last_int in {"joke","anecdote","fun_fact","fact"}:
            pool=[r for r in INTENTS[last_int]["responses"] if r!=last_bot]
            return random.choice(pool) if pool else random.choice(INTENTS[last_int]["responses"])
        if last_int in {"music","movie","game","series"}:
            return recommend(last_int, prefs.get(f"{last_int}_genre"))

    # 15. крестики-нолики
    if "крестики" in low:
        user_data["tic_tac_toe"]=TicTacToe()
        return "Начинаем «крестики-нолики»!\n"+user_data['tic_tac_toe'].render()+"\nТвой ход (A1..C3):"

    # 16. жанр после follow-up
    for cat in {"music","movie","game","series"}:
        if last_int==cat and f"{cat}_genre" not in prefs:
            rec=recommend(cat, text.strip()); prefs[f"{cat}_genre"]=text.strip()
            user_data["last_bot"]=rec
            return rec

    # 17. sentiment + intent-predict
    lemma = lemmatize_text(
        " ".join(correct_spelling(w, DICTIONARY) for w in clean_text(text).split())
    )
    tone = "Мне жаль, что тебе грустно. " if get_sentiment(lemma)<-0.2 else \
           "Рад за тебя! "                if get_sentiment(lemma)>0.5  else ""

    intent=None
    for p in (lambda x: clf.predict(x), lambda x: clf.predict_fuzzy(x)):
        try:
            c=p(lemma)
            if c in INTENTS: intent=c; break
        except: pass

    if intent in {"music","movie","game","series"}:
        user_data.update(last_intent=intent, asked_followup=True, awaiting_genre=intent)
        return INTENTS[intent]["follow_up"][0]

    if intent:
        opts=INTENTS[intent]["responses"]
        if last_bot in opts and len(opts)>1: opts=[o for o in opts if o!=last_bot]
        resp=random.choice(opts)
        user_data["last_bot"]=resp
        if not user_data["asked_followup"]:
            for f in INTENTS[intent].get("follow_up", []):
                if f not in user_data["asked_questions"]:
                    resp+=" "+f
                    user_data["asked_questions"].add(f)
                    user_data["asked_followup"]=True
                    break
        user_data["last_intent"]=intent
        return tone+resp

    # 18. retrieval-ответ
    cand=retriever.get_answer(lemma)
    if cand:
        user_data.update(last_bot=cand,last_intent=None)
        return tone+cand

    # 19. fallback → обучение
    cid=f'c{re.sub(r"[^a-z0-9]","",low_clean) or "intent"}'
    new_i={"examples":[text],"responses":["Я пока не знаю, как на это ответить. Подскажите пример ответа?"]}
    extra=json.loads(CUSTOM_F.read_text('utf-8')) if CUSTOM_F.exists() else {}
    extra[cid]=new_i; _save_custom_intents(extra); INTENTS[cid]=new_i
    user_data["awaiting_teach"]=text
    return "Я пока не знаю, как на это отвечать. Подскажите пример ответа?"

# ────────── helper: отправить voice-сообщение ───────────────────────────────
def _reply_voice(update: Update, reply_text: str, stub: str):
    wav = TEMP_DIR / f"{stub}.wav"
    ogg = TEMP_DIR / f"{stub}.ogg"
    _tts_to_wav(reply_text, wav)
    AudioSegment.from_wav(wav).export(ogg, format="ogg", codec="libopus", bitrate="48k")
    with open(ogg, "rb") as f:
        update.message.reply_voice(voice=f, caption=reply_text)

# ────────── Telegram-handlers ───────────────────────────────────────────────
def handle_voice(update: Update, context: CallbackContext):
    au = update.message.voice or update.message.audio or update.message.document
    if not au:
        return update.message.reply_text("Не смог получить аудио.")
    fid = getattr(au, "file_unique_id", None) or update.message.message_id
    src = TEMP_DIR / f"{fid}.src"
    wav_in = TEMP_DIR / f"{fid}.in.wav"
    au.get_file().download(str(src))
    # корректная конверсия: 48k/opus → 16k 16-bit mono, +6 dB
    audio = AudioSegment.from_file(src)
    audio = (audio.set_frame_rate(16000)
                   .set_channels(1)
                   .set_sample_width(2)   # 16-bit
                   .apply_gain(+6))       # чуточку громче
    audio.export(wav_in, format="wav")

    try:
        user_text = stt_from_wav(str(wav_in))
    except Exception as e:
        return update.message.reply_text(f"Ошибка распознавания: {e}")

    # ⬇️ Новая проверка
    if not user_text.strip():
        return update.message.reply_text(
            "Извините, не расслышал – попробуйте ещё раз произнести чуть отчётливее."
        )

    update.message.reply_text(f"🗣 Вы сказали: {user_text}")

    ud=context.user_data; hist=ud.setdefault("history",deque(maxlen=50))
    bot_text=get_response(user_text, ud, hist); hist.extend((user_text, bot_text))
    _reply_voice(update, bot_text, f"{fid}_resp")

def handle_text(update: Update, context: CallbackContext):
    user_text = update.message.text
    update.message.reply_text(f"🗣 Вы сказали: {user_text}")
    ud=context.user_data; hist=ud.setdefault("history",deque(maxlen=50))
    bot_text=get_response(user_text, ud, hist); hist.extend((user_text, bot_text))
    _reply_voice(update, bot_text, f"{update.message.message_id}_resp")

def start(update: Update,_): update.message.reply_text("Привет! Пришлите текст или голос — отвечу голосом 🙂")
def help_command(update: Update,_): update.message.reply_text("Я распознаю речь (Vosk) и отвечаю voice-сообщением.")

# ────────── main() ─────────────────────────────────────────────────────────-
def main():
    token=os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN не задан.")
    up=Updater(token)
    dp=up.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help",  help_command))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    dp.add_handler(MessageHandler(Filters.voice | Filters.audio | Filters.document, handle_voice))
    up.start_polling(); up.idle()

if __name__=="__main__":
    main()
