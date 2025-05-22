# recommendations.py

import random
import re

# Небольшие списки для примера
MOVIE_RECS = {
    "комедия": [
        "«1+1» (Intouchables)",
        "«Очень плохие мамочки»",
        "«Ночи в стиле буги»"
    ],
    "драма": [
        "«Зелёная миля»",
        "«Титаник»",
        "«Побег из Шоушенка»"
    ],
    "фантастика": [
        "«Интерстеллар»",
        "«Начало»",
        "«Матрица»"
    ],
    "боевик": [
        "«Джон Уик» (John Wick)",
        "«Безумный Макс: Дорога ярости»",
        "«Крепкий орешек»"
    ],
    "триллер": [
        "«Семь» (Se7en)",
        "«Исчезнувшая» (Gone Girl)",
        "«Остров проклятых» (Shutter Island)"
    ],
    "ужасы": [
        "«Заклятие» (The Conjuring)",
        "«Прочь» (Get Out)",
        "«Оно» (It)"
    ],
    "анимация": [
        "«Унесённые призраками» (Spirited Away)",
        "«История игрушек» (Toy Story)",
        "«Как приручить дракона»"
    ],
    "документальный": [
        "«Free Solo»",
        "«Последний танец»",
        "«Затонувший корабль»"
    ],
    "приключения": [
        "«Индиана Джонс: В поисках утраченного ковчега»",
        "«Парк Юрского периода»",
        "«Властелин колец: Братство кольца»"
    ],
    "криминал": [
        "«Крёстный отец»",
        "«Криминальное чтиво»",
        "«Славные парни»"
    ],
    "романтика": [
        "«Дневник» (The Notebook)",
        "«Ла-Ла Ленд»",
        "«Гордость и предубеждение»"
    ],
    "военный": [
        "«Спасти рядового Райана»",
        "«Цельнометаллическая оболочка»",
        "«Дюнкерк»"
    ],
    "исторический": [
        "«Гладиатор»",
        "«Король говорит!»",
        "«Храброе сердце»"
    ],
    "семейный": [
        "«Суперсемейка»",
        "«Паддингтон»",
        "«Матильда»"
    ]
}

MUSIC_RECS = {
    "рок": [
        "Queen — Bohemian Rhapsody",
        "Nirvana — Smells Like Teen Spirit",
        "AC/DC — Thunderstruck"
    ],
    "джаз": [
        "Miles Davis — So What",
        "John Coltrane — Naima",
        "Billie Holiday — Strange Fruit"
    ],
    "поп": [
        "Taylor Swift — Love Story",
        "Ariana Grande — 7 rings",
        "Ed Sheeran — Shape of You"
    ],
    "классика": [
        "Ludwig van Beethoven — Symphony No.5",
        "Wolfgang Amadeus Mozart — Eine kleine Nachtmusik",
        "Pyotr Ilyich Tchaikovsky — Swan Lake"
    ],
    "хип-хоп": [
        "Kendrick Lamar — HUMBLE.",
        "Eminem — Lose Yourself",
        "Travis Scott — Sicko Mode"
    ],
    "электронная": [
        "Daft Punk — One More Time",
        "Avicii — Levels",
        "Calvin Harris — Summer"
    ],
    "блюз": [
        "B.B. King — The Thrill Is Gone",
        "Muddy Waters — Hoochie Coochie Man",
        "Etta James — I'd Rather Go Blind"
    ],
    "кантри": [
        "Johnny Cash — Ring of Fire",
        "Dolly Parton — Jolene",
        "Luke Combs — Beautiful Crazy"
    ],
    "регги": [
        "Bob Marley — No Woman, No Cry",
        "Peter Tosh — Legalize It",
        "Jimmy Cliff — The Harder They Come"
    ],
    "фолк": [
        "Bob Dylan — Blowin’ in the Wind",
        "Simon & Garfunkel — The Sound of Silence",
        "Joni Mitchell — Big Yellow Taxi"
    ],
    "металл": [
        "Metallica — Enter Sandman",
        "Iron Maiden — The Trooper",
        "Black Sabbath — Paranoid"
    ],
    "панк": [
        "Sex Pistols — Anarchy in the UK",
        "The Ramones — Blitzkrieg Bop",
        "Green Day — Basket Case"
    ],
    "R&B": [
        "Beyoncé — Halo",
        "Usher — U Got It Bad",
        "The Weeknd — Blinding Lights"
    ],
    "соул": [
        "Otis Redding — (Sittin’ On) the Dock of the Bay",
        "Aretha Franklin — Respect",
        "Marvin Gaye — What’s Going On"
    ]
}

GAME_RECS = {
    "стратегия": [
        "Civilization VI",
        "Age of Empires II",
        "StarCraft II"
    ],
    "шутер": [
        "DOOM Eternal",
        "Counter-Strike: Global Offensive",
        "Call of Duty: Modern Warfare"
    ],
    "рпг": [
        "The Witcher 3",
        "Skyrim",
        "Cyberpunk 2077"
    ],
    "приключения": [
        "The Legend of Zelda: Breath of the Wild",
        "Uncharted 4",
        "Tomb Raider"
    ],
    "головоломка": [
        "Portal 2",
        "The Witness",
        "Myst"
    ],
    "выживание": [
        "Minecraft",
        "Ark: Survival Evolved",
        "Subnautica"
    ],
    "гоночные": [
        "Forza Horizon 5",
        "Mario Kart 8 Deluxe",
        "Need for Speed Heat"
    ],
    "спортивные": [
        "FIFA 22",
        "NBA 2K22",
        "Madden NFL 22"
    ],
    "симулятор": [
        "The Sims 4",
        "Microsoft Flight Simulator",
        "Cities: Skylines"
    ],
    "платформер": [
        "Super Mario Odyssey",
        "Celeste",
        "Hollow Knight"
    ],
    "файтинг": [
        "Street Fighter V",
        "Tekken 7",
        "Mortal Kombat 11"
    ],
    "MOBA": [
        "League of Legends",
        "Dota 2",
        "Heroes of the Storm"
    ]
}

SERIES_RECS = {
    "триллер": [
        "«Во все тяжкие»",
        "«Шерлок»",
        "«Острые козырьки»"
    ],
    "фэнтези": [
        "«Игра престолов»",
        "«Ведьмак»",
        "«Однажды в сказке»"
    ],
    "драма": [
        "«Это мы»",
        "«Благие знамения»",
        "«Карточный домик»"
    ],
    "комедия": [
        "Friends",
        "The Office",
        "The Big Bang Theory"
    ],
    "криминал": [
        "Narcos",
        "Mindhunter",
        "True Detective"
    ],
    "научная фантастика": [
        "Black Mirror",
        "Westworld",
        "Stranger Things"
    ],
    "ужасы": [
        "The Haunting of Hill House",
        "American Horror Story",
        "Penny Dreadful"
    ],
    "документальные": [
        "Planet Earth",
        "Making a Murderer",
        "The Last Dance"
    ],
    "реалити": [
        "Survivor",
        "The Great British Bake Off",
        "Keeping Up with the Kardashians"
    ],
    "аниме": [
        "Attack on Titan",
        "Naruto",
        "Death Note"
    ],
    "романтика": [
        "Outlander",
        "Normal People",
        "Bridgerton"
    ],
    "исторические": [
        "The Crown",
        "Vikings",
        "Chernobyl"
    ],
    "мистика": [
        "Twin Peaks",
        "Dark",
        "The X-Files"
    ],
    "супергерои": [
        "Daredevil",
        "The Boys",
        "WandaVision"
    ],
    "мультсериал": [
        "Rick and Morty",
        "BoJack Horseman",
        "Avatar: The Last Airbender"
    ]
}

def recommend(category: str, genre: str) -> str:
    """
    Возвращает 1–2 случайных рекомендации по заданной категории
    (movie/music/game/series) и жанру.
    Если жанр не найден — уведомляет об этом.
    """
    # очистка ответа пользователя от «да,», «ага,», «конечно» и перевод в lower()
    genre_clean = re.sub(
        r'^(да[, ]+|ага[, ]+|конечно[, ]+)',
        '', genre.strip(),
        flags=re.IGNORECASE
    ).lower().strip()

    # словарь по категориям
    data = {
        "movie": MOVIE_RECS,
        "music": MUSIC_RECS,
        "game":   GAME_RECS,
        "series": SERIES_RECS
    }.get(category, {})

    # если вообще не указал жанр
    if not genre_clean:
        available = ", ".join(sorted(data.keys()))
        return f"Какой жанр тебе интересен? Доступные жанры: {available}."

    # ищем список по жанру
    lst = data.get(genre_clean)
    if not lst:
        available = ", ".join(sorted(data.keys()))
        return f"Не распознал жанр «{genre_clean}». Доступные: {available}."

    # выбираем 1–2 рекомендации
    picks = random.sample(lst, min(2, len(lst)))
    return f"Вот что я могу порекомендовать в жанре «{genre_clean}»: " + "; ".join(picks)