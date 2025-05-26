import os
from copy import deepcopy
from pathlib import Path
from telegram import Update
from telegram.ext import (Updater, CommandHandler, MessageHandler,
                          Filters, CallbackContext)

from bot_logic      import get_response
from file_memory    import load_history, save_history, load_user_data, save_user_data, MEM_DIR
from modules.tictactoe import TicTacToe

# ───────────────────────── plugins ──────────────────────────
from modules.help_module      import register_handlers as help_reg
from modules.smalltalk_module import register_handlers as smalltalk_reg
from modules.settings_module  import register_handlers as settings_reg
from modules.catalog_module   import register_handlers as catalog_reg
from modules.reminder_module  import register_handlers as reminder_reg
# ────────────────────────────────────────────────────────────

TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise SystemExit("Переменная TELEGRAM_TOKEN не задана")

# ────────────────────────── handlers ────────────────────────
def start(update: Update, context: CallbackContext) -> None:
    """/start — приветствие и полный сброс памяти пользователя."""
    update.message.reply_text("Привет! Я Альфред, твой бот-собеседник. Спрашивай что угодно!")
    uid = update.effective_user.id
    context.user_data.clear()                         # RAM
    (MEM_DIR / f"history_{uid}.json").unlink(missing_ok=True)
    (MEM_DIR / f"user_data_{uid}.json").unlink(missing_ok=True)

def handle_message(update: Update, context: CallbackContext) -> None:
    uid, text = update.effective_user.id, update.message.text

    # ➊ — сохраняем все «несериализуемые» объекты, чтобы не потерять
    volatile_objects = {k: v for k, v in context.user_data.items()
                        if isinstance(v, TicTacToe)}          # при желании добавить и другие типы

    # ➋ — поднимаем долгую память
    history      = load_history(uid)
    stored_state = load_user_data(uid)

    context.user_data.clear()
    context.user_data.update(stored_state)
    context.user_data.update(volatile_objects)      # вернули RAM-объекты

    # ➌ — крестики-нолики
    if isinstance(context.user_data.get("tic_tac_toe"), TicTacToe):
        game: TicTacToe = context.user_data["tic_tac_toe"]
        result, finished = game.player_move(text)
        update.message.reply_text(result)
        if finished:
            context.user_data.pop("tic_tac_toe", None)
        return

    # ➍ — teach-on-the-fly
    if "awaiting_teach" in context.user_data:
        pattern = context.user_data.pop("awaiting_teach")
        context.user_data.setdefault("custom_answers", {})[pattern] = text
        update.message.reply_text("Спасибо! Я запомнил твой пример ответа 🙂")
        history.append(text)
        save_history(uid, history)
        _safe_save(uid, context.user_data)
        return

    # ➎ — основная логика
    reply = get_response(text, context.user_data, history)
    update.message.reply_text(reply)

    history.append(text)
    save_history(uid, history)
    _safe_save(uid, context.user_data)

# ——— помощник: удаляем несериализуемые объекты перед сохранением ———
def _safe_save(uid: int, state: dict) -> None:
    safe = deepcopy(state)
    safe.pop("tic_tac_toe", None)
    save_user_data(uid, safe)
# ────────────────────────────────────────────────────────────

def main() -> None:
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    # плагины
    help_reg(dp); smalltalk_reg(dp); settings_reg(dp)
    catalog_reg(dp); reminder_reg(dp)

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message), group=100)

    # выключаем webhook → можем использовать long-polling
    updater.bot.delete_webhook(drop_pending_updates=True)
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
