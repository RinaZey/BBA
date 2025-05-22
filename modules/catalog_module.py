# modules/catalog_module.py

import json
from pathlib import Path
from telegram.ext import CommandHandler

# Путь к вашему каталогу (по умолчанию data/catalog.json)
DATA_DIR     = Path(__file__).parent.parent / "data"
CATALOG_FILE = DATA_DIR / "catalog.json"

# Пытаемся загрузить каталог, если его нет — используем пустой словарь
try:
    with open(CATALOG_FILE, encoding="utf-8") as f:
        CATALOG = json.load(f)
except FileNotFoundError:
    CATALOG = {}

def show_catalog(update, context):
    """
    /catalog — покажем первые 3 категории из каталога,
    если каталог не загружен — сообщаем об этом.
    """
    if not CATALOG:
        update.message.reply_text("Извините, каталог временно недоступен.")
        return

    # допустим, в CATALOG хранится словарь {category: [item, ...], ...}
    lines = []
    for cat, items in CATALOG.items():
        # берем по одному элементу из первых трёх категорий
        sample = items[:1]  # или [:3] если хотите сразу по 3 товара из каждой
        for it in sample:
            name = it.get("name", "–")
            price = it.get("price", "–")
            lines.append(f"{cat.capitalize()}: {name} — {price}₽")
        if len(lines) >= 3:
            break

    update.message.reply_text("Каталог:\n" + "\n".join(lines))

def register_handlers(dp):
    dp.add_handler(CommandHandler("catalog", show_catalog))
