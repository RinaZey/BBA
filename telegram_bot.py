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
    context.user_data.clear()

def handle_message(update: Update, context: CallbackContext) -> None:
    text = update.message.text

    # —————— Обучение на лету: если ждали teach-пароля ——————
    if 'awaiting_teach' in context.user_data:
        key = context.user_data.pop('awaiting_teach')
        # сохраняем пользовательский ответ
        context.user_data.setdefault('custom_answers', {})[key] = text
        update.message.reply_text("Спасибо! Я запомнил(а) твой ответ.")
        return

    # обычная логика
    resp = get_response(text, context.user_data)
    update.message.reply_text(resp)

def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
