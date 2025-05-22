import json
from pathlib import Path
from collections import deque

# где будем хранить файлы
BASE_DIR = Path(__file__).parent
MEM_DIR = BASE_DIR / 'user_memory'
MEM_DIR.mkdir(exist_ok=True)

def load_history(user_id: int, maxlen: int = 50) -> deque:
    """
    Загружает последние maxlen сообщений пользователя из файла history_{user_id}.json
    """
    path = MEM_DIR / f'history_{user_id}.json'
    if not path.exists():
        return deque(maxlen=maxlen)
    data = json.loads(path.read_text(encoding='utf-8'))
    return deque(data[-maxlen:], maxlen=maxlen)

def save_history(user_id: int, history: deque) -> None:
    """
    Сохраняет всю историю (или последние maxlen) пользователя в файл
    """
    path = MEM_DIR / f'history_{user_id}.json'
    path.write_text(
        json.dumps(list(history), ensure_ascii=False),
        encoding='utf-8'
    )

def _serialize(obj):
    """
    Рекурсивно преобразует deque->list, set->list, пропуская несериализуемые объекты.
    """
    if isinstance(obj, deque):
        return [_serialize(i) for i in obj]
    if isinstance(obj, set):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[k] = _serialize(v)
            except Exception:
                # пропускаем несериализуемые
                continue
        return out
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    # базовые типы
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # всё остальное — приводим к строке
    try:
        return str(obj)
    except Exception:
        return None

def load_user_data(user_id: int) -> dict:
    """
    Загружает контекст user_data из файла user_data_{user_id}.json
    """
    path = MEM_DIR / f'user_data_{user_id}.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))

def save_user_data(user_id: int, user_data: dict) -> None:
    """
    Сохраняет контекст user_data в файл, предварительно сериализовав все структуры.
    """
    path = MEM_DIR / f'user_data_{user_id}.json'
    clean = _serialize(user_data)
    path.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
