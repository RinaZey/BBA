from telegram.ext import CommandHandler

def cmd_commands(update, context):
    text = (
        "/commands — список команд\n"
        "/help — краткая справка\n"
        "/settings — твои настройки\n"
        "/tictactoe — игра крестики-нолики\n"
        "/catalog — посмотреть каталог\n"
        "/remind <м> <текст> — напомнить через м минут\n"
        "...и многое другое!"
    )
    update.message.reply_text(text)

def cmd_settings(update, context):
    ud = context.user_data
    lang = ud.get("lang", "ru")
    tz   = ud.get("tz", "UTC")
    freq = ud.get("reminder_freq", "—")
    text = f"Твои настройки:\n• Язык: {lang}\n• Часовой пояс: {tz}\n• Напоминания: {freq}"
    update.message.reply_text(text)

def register_handlers(dp):
    dp.add_handler(CommandHandler("commands", cmd_commands))
    dp.add_handler(CommandHandler("help",     cmd_commands))
    dp.add_handler(CommandHandler("settings", cmd_settings))
