<p align="center">
  <img
    src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=28&duration=4000&pause=800&color=00C9FF&center=true&vCenter=true&width=900&height=100&lines=BBA+-+Bot+Based+Assistant;NLP-powered+Telegram+assistant;Machine-learning+core+%2B+rich+features"
    alt="BBA animated headline"
  />
</p>

<p align="center">
  <img src="https://img.shields.io/github/languages/top/RinaZey/BBA?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/github/last-commit/RinaZey/BBA?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Telegram%20Bot%20API-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <img src="docs/demo.gif" width="700" alt="Bot demo" />
</p>

---

## О чём проект

**BBA (Bed Bot Alfred)** — многофункциональный Telegram-бот, который совмещает NLP-обработку и машинное обучение. Он может стать не только приятным собеседником, но и поиграть с тобой в крестики-нолики, а также помочь с выбором нескольких кроватей и матрассов ;)

<details>
  <summary>Ключевые возможности</summary>

- **Интеллектуальный классификатор интентов** (TF-IDF / BERT + SVM)  
- **Обработка естественного языка**: очистка, лемматизация, коррекция опечаток  
- **STT / TTS**: преобразование речи в текст  
- **Seq-to-Seq / LLM-интеграция** для генерации реплик  
- **Игровые модули**: крестики-нолики 
- **Каталог товаров** 
- **AI-обработка изображений** (фильтры, наложения, OCR)  
- **Модуль токсичности** и фильтр запрещённых тем  
- **Отчёты и аналитика** (журналы, графики обучения, confusion-matrix)  
</details>

---

## Стек & инструменты

<p align="center">
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"       height="55" alt="Python"/>
  <img src="https://github.com/devicons/devicon/blob/master/icons/pytorch/pytorch-original.svg"     height="55" alt="PyTorch"/>
  <img src="https://github.com/devicons/devicon/blob/master/icons/scikitlearn/scikitlearn-original.svg" height="55" alt="Scikit-Learn"/>
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/telegram.svg" height="55" alt="Telegram"/>
</p>

---

## Структура репозитория

```text
BBA/
├── bot_logic.py           # Главный «мозг» бота
├── nlp_utils/             # Чистка, лемматизация, коррекция текста
│   ├── cleaning.py
│   └── ...
├── intent_classifier/     # ML-классификатор + тренировка
├── modules/               # Игровые и вспомогательные модули
└── data/                  # Датасеты, модели, каталоги
```

---

## Быстрый старт

1. **Подготовка к запуску**

   ```bash
   git clone https://github.com/RinaZey/BBA.git
   cd BBA
   ```

2. **Локальный запуск**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   export TELEGRAM_TOKEN=<YOUR_TOKEN>
   python bot.py
   ```
---

## Контрибьютинг

Буду рада PR-ам и идеям: баг-фиксы, новые интенты, улучшения модели.  
Перед отправкой PR выполните:

```bash
# проверка типов + стиль
mypy .
ruff .
pytest
```
