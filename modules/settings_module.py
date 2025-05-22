# modules/settings_module.py

from telegram.ext import CommandHandler, MessageHandler, Filters

def cmd_settings(update, context):
    """
    /settings — запускает диалог настройки уведомлений
    """
    update.message.reply_text(
        "Я помогу настроить всё под тебя.\n"
        "Отключить ли ежедневные уведомления? (да/нет)"
    )
    context.user_data["awaiting_setting"] = "toggle_notifications"

def handle_yesno(update, context):
    """
    Перехватывает ответ 'да'/'нет' после вопроса о настройках.
    Если не в состоянии ожидания — сразу возвращает None, чтобы дальше шел smalltalk или fallback.
    """
    state = context.user_data.get("awaiting_setting")
    if state != "toggle_notifications":
        return  # не наша ситуация — передаём дальше

    text = update.message.text.strip().lower()
    if text in ("да", "yes", "y"):
        context.user_data["daily_notifications"] = False
        update.message.reply_text("Окей, ежедневные уведомления отключены.")
    elif text in ("нет", "no", "n"):
        context.user_data["daily_notifications"] = True
        update.message.reply_text("Хорошо, ежедневные уведомления оставлены включёнными.")
    else:
        return update.message.reply_text("Пожалуйста, ответь “да” или “нет”.")
    # сброс состояния
    context.user_data.pop("awaiting_setting")

def register_handlers(dp):
    # Команда /settings
    dp.add_handler(CommandHandler("settings", cmd_settings))
    # Ответы да/нет должны отработать раньше smalltalk_module
    dp.add_handler(
        MessageHandler(Filters.text & ~Filters.command, handle_yesno),
        group=0
    )
