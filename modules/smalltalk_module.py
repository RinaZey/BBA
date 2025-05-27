from telegram.ext import MessageHandler, Filters

TOXIC = {"дурак", "тупой", "убью", "сдохни", "ты дурак", "пошел ты", "ты идиот", "бесишь тварь", "ты долбаный дятел", "мудак", "какой ты дебил", "иди в жопу", "отвали идиот", "долбоеб", "иди на хрен", "ты тупой", "лох", "ублюдок", "выблядок"}
MALICIOUS = {"взрыв", "убить", "бомба"}    # запрещённые темы

def smalltalk_handler(update, context):
    text = update.message.text.lower()
    for w in TOXIC:
        if w in text:
            return update.message.reply_text("Пожалуйста, без оскорблений.")
    for w in MALICIOUS:
        if w in text:
            return update.message.reply_text("Извини, в этом я помочь не могу.")

def register_handlers(dp):
    # group=0, чтобы первыми заходили фильтры
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, smalltalk_handler), group=0)
