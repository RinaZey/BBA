from telegram.ext import CommandHandler
from datetime import datetime, timedelta

def remind_cmd(update, context):
    try:
        mins = int(context.args[0])
        text = " ".join(context.args[1:])
    except:
        return update.message.reply_text("Используй: /remind <минут> <текст>")
    run_at = datetime.now() + timedelta(minutes=mins)
    context.job_queue.run_once(
        lambda ctx: ctx.bot.send_message(update.effective_chat.id, text),
        when=run_at
    )
    update.message.reply_text(f"Напомню через {mins} мин.")

def register_handlers(dp):
    dp.add_handler(CommandHandler("remind", remind_cmd))
