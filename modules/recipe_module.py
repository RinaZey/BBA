from telegram.ext import CommandHandler

LOCAL_RECIPES = {
    "борщ": "Борщ: свёклу, капусту, мясо, варить 2 часа…",
    "мафины": "Маффины: смесь, выпекать 20 мин при 180°C…",
}

def recipe_cmd(update, context):
    key = " ".join(context.args).lower()
    ans = LOCAL_RECIPES.get(key, "Простого рецепта не нашёл, уточни запрос.")
    update.message.reply_text(ans)

def register_handlers(dp):
    dp.add_handler(CommandHandler("recipe", recipe_cmd))
