# telegram_bot.py

import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from bot_logic import get_response

TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TOKEN:
    print("TELEGRAM_TOKEN не задан. Задайте переменную окружения TELEGRAM_TOKEN.")
    exit(1)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я Альфред, твой бот-собеседник. Спрашивай что угодно!')

def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Напиши мне сообщение, а я постараюсь ответить :)')

def handle_message(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    # Теперь get_response возвращает только строку
    response = get_response(user_text)
    update.message.reply_text(response)

def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
