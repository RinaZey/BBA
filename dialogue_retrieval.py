# dialogue_retrieval.py
import random
from pathlib import Path
from typing import List, Tuple, Callable

from nltk.metrics import edit_distance


class DialogueRetriever:
    """
    Простое «поиск-ответ» по корпусу диалогов.

    • Файл-корпус — текст, в котором каждая пара «вопрос/ответ» отделена
      пустой строкой. Первой строчкой блока считается вопрос, второй — ответ.

    • Метод get_answer(query)          — основной.
    • Метод reply(query)               — 100-%-й алиас для обратной совместимости.
    """
    #: при каком нормализованном расстоянии считаем пару релевантной
    DEFAULT_THRESHOLD: float = 0.40

    def __init__(
        self,
        filepath: str | Path,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        self.pairs: List[Tuple[str, str]] = []
        self.preprocess = preprocess or (lambda s: s.lower())

        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Dialogue corpus not found: {path}")

        with path.open(encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")

        for block in blocks:
            q, *rest = block.strip().splitlines()
            if rest:                                      # есть ответ
                self.pairs.append((self.preprocess(q),    # храним уже «очищенный» вопрос
                                   rest[0].strip()))

        if not self.pairs:
            raise ValueError("Dialogue corpus is empty or malformed")

    # ──────────────────────────────────────────────────────
    def _norm_dist(self, s1: str, s2: str) -> float:
        return edit_distance(s1, s2) / max(1, len(s2))

    # ──────────────────────────────────────────────────────
    def get_answer(self, query: str, threshold: float | None = None) -> str | None:
        """
        Возвращает наиболее подходящий ответ, если нормализованное
        расстояние Левенштейна < threshold. Иначе — None.
        """
        if not query.strip():
            return None

        q_prep   = self.preprocess(query)
        thr      = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        best_q, best_a = min(
            self.pairs,
            key=lambda qa: self._norm_dist(q_prep, qa[0])
        )
        if self._norm_dist(q_prep, best_q) < thr:
            return best_a
        return None

    # ──────────────────────────────────────────────────────
    # alias, чтобы старый вызов retriever.reply(...) продолжал работать
    reply = get_answer
