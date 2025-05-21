# telegram_bot.py

import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from bot_logic import get_response

TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TOKEN:
    print("TELEGRAM_TOKEN не задан.")
    exit(1)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я Альфред, твой бот-собеседник. Спрашивай что угодно!')
    context.user_data.clear()  # сбросить контекст для нового чата

def handle_message(update: Update, context: CallbackContext) -> None:
    text = update.message.text

    # 1) сначала проверка на уточняющий вопрос для day
    last = context.user_data.get('last_intent')
    low = text.lower()
    if last == 'day' and 'как именно' in low:
        # даём специализированный ответ без ML
        resp = random.choice(INTENTS['day_details']['responses'])
        # продолжаем хранить last_intent как day_details
        context.user_data['last_intent'] = 'day_details'
        update.message.reply_text(resp)
        return

    # 2) во всех остальных случаях — обычный get_response
    response, intent = get_response(text)
    # сохраняем интент в контекст
    context.user_data['last_intent'] = intent
    update.message.reply_text(response)

def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
