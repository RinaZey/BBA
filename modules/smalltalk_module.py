from telegram.ext import MessageHandler, Filters, DispatcherHandlerStop

TOXIC = {"дурак", "тупой", "убью", "сдохни", "ты дурак", "пошел ты", "ты идиот", "бесишь тварь", "ты долбаный дятел", "мудак", "какой ты дебил", "иди в жопу", "отвали идиот", "долбоеб", "иди на хрен", "ты тупой", "лох", "ублюдок", "выблядок", "придурок", "глупый", "уёбок", "шлюха", "сын шлюхи", "дебил", "дибил"}
MALICIOUS = {"взрыв", "убить", "бомба", "сделать взрыв", "убийство"}    # запрещённые темы

def smalltalk_filter(update, context):
    text = (update.message.text or "").lower()

    if any(w in text for w in TOXIC):
        update.message.reply_text("Пожалуйста, без оскорблений.")
        raise DispatcherHandlerStop()     # ← прерываем цепочку

    if any(w in text for w in MALICIOUS):
        update.message.reply_text("Извини, в этом я помочь не могу.")
        raise DispatcherHandlerStop()     # ← стоп!

    # если всё нормально – ничего не делаем, просто выходим

def register_handlers(dp):
    # фильтр первым (group=0), чтобы мог остановить дальнейшие хэндлеры
    dp.add_handler(
        MessageHandler(Filters.text & ~Filters.command, smalltalk_filter),
        group=0
    )