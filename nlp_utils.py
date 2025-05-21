# nlp_utils.py
import re
from Levenshtein import distance as lev_distance
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc

# инициализация Natasha
_segmenter = Segmenter()
_emb = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)
_morph_vocab = MorphVocab()

def clean_text(text: str) -> str:
    """
    Приводим к нижнему регистру, удаляем лишние символы и повторяющиеся пробелы.
    """
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def correct_spelling(word: str, dictionary: set, max_dist: int = 2) -> str:
    """
    Проверяем слово в словаре; если нет — ищем ближайшее по Левенштейну.
    """
    if word in dictionary:
        return word
    # ищем кандидатов с допустимой dist
    candidates = [w for w in dictionary if abs(len(w) - len(word)) <= max_dist]
    best = min(candidates, key=lambda w: lev_distance(w, word), default=word)
    return best if lev_distance(best, word) <= max_dist else word

def lemmatize_text(text: str) -> str:
    """
    Возвращает исходную фразу, преобразованную в леммы.
    """
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    for token in doc.tokens:
        token.lemmatize(_morph_vocab)
    return ' '.join([t.lemma for t in doc.tokens])
