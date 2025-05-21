# dialogue_retrieval.py
import random
from nltk import edit_distance

class DialogueRetriever:
    def __init__(self, filepath: str):
        # загружаем пары вопрос–ответ
        self.pairs = []
        with open(filepath, encoding='utf-8') as f:
            content = f.read().split('\n\n')
        for block in content:
            lines = block.split('\n')
            if len(lines) >= 2:
                q, a = lines[0].strip(), lines[1].strip()
                self.pairs.append((q, a))

    def get_answer(self, query: str, threshold: float = 0.4) -> str:
        """
        Ищет в self.pairs наиболее близкий вопрос по edit_distance;
        возвращает ответ, если нормализованное расстояние < threshold.
        """
        best = min(self.pairs, key=lambda qa: edit_distance(query, qa[0]) / max(1, len(qa[0])))
        norm_dist = edit_distance(query, best[0]) / max(1, len(best[0]))
        if norm_dist < threshold:
            return best[1]
        return None
