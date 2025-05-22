import os
from pathlib import Path
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from bot_logic import get_response
from file_memory import (
    load_history, save_history,
    load_user_data, save_user_data,  # ▼▼▼ теперь используем
    MEM_DIR
)

# импортируем модули
from modules.help_module      import register_handlers as help_reg
from modules.settings_module  import register_handlers as settings_reg
from modules.smalltalk_module import register_handlers as smalltalk_reg
from modules.catalog_module   import register_handlers as catalog_reg
from modules.reminder_module  import register_handlers as reminder_reg
from modules.recipe_module    import register_handlers as recipe_reg
from modules.tictactoe        import TicTacToe

TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    print("TELEGRAM_TOKEN не задан.")
    exit(1)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        'Привет! Я Альфред, твой бот-собеседник. Спрашивай что угодно!'
    )
    # — очистить context и файлы на диске
    context.user_data.clear()
    user_id = update.effective_user.id

    # удалить файл-историю
    hist_file = MEM_DIR / f'history_{user_id}.json'
    if hist_file.exists():
        hist_file.unlink()

    # удалить сохранённый user_data
    data_file = MEM_DIR / f'user_data_{user_id}.json'
    if data_file.exists():
        data_file.unlink()


def handle_message(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    text = update.message.text

    # — загрузить историю и user_data из файлов
    history     = load_history(user_id)
    stored_data = load_user_data(user_id)
    context.user_data.clear()
    context.user_data.update(stored_data)

    # — если крестики-нолики запущены — обрабатываем ход
    if 'tic_tac_toe' in context.user_data:
        game: TicTacToe = context.user_data['tic_tac_toe']
        result, finished = game.player_move(text)
        update.message.reply_text(result)
        if finished:
            del context.user_data['tic_tac_toe']
        return

    # — teach-on-the-fly
    if 'awaiting_teach' in context.user_data:
        key = context.user_data.pop("awaiting_teach")
        context.user_data.setdefault("custom_answers", {})[key] = text
        update.message.reply_text("Спасибо! Я запомнил(а) твой ответ.")
        history.append(text)
        save_history(user_id, history)
        save_user_data(user_id, context.user_data)  # ▼▼▼ сохраняем сразу после teach
        return

    # — основной NLU-fallback
    resp = get_response(text, context.user_data, history)
    update.message.reply_text(resp)

    # — обновляем историю и сохраняем
    history.append(text)
    save_history(user_id, history)

    # — сохраняем весь user_data на диск
    save_user_data(user_id, context.user_data)


def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher

    # 1) Справка и настройки
    help_reg(dp)
    # 2) Интерактивные настройки
    settings_reg(dp)
    # 3) Small-talk, фильтр токсичности и malicious
    smalltalk_reg(dp)
    # 4) Каталог и продажи
    catalog_reg(dp)
    # 5) Напоминания
    reminder_reg(dp)
    # 6) Рецепты
    recipe_reg(dp)

    # команда /start
    dp.add_handler(CommandHandler("start", start))

    # общий текстовый fallback
    dp.add_handler(
        MessageHandler(Filters.text & ~Filters.command, handle_message),
        group=100
    )

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
