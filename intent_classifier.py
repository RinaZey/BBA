"""
intent_classifier.py
────────────────────
Обучает классификатор интентов.

• Признаки = TF-IDF( word 1-2-gram  ∪  char 3-5-gram, min_df=3 для char )
• Модель   = SGDClassifier(log_loss, early_stopping) с class_weight
• Во время обучения рисует:
    – history (loss / acc)
    – learning curve
    – confusion-matrix (батчи ≤15 классов)
    – elbow-plot для K-Means
Сохраняет:
    intent_clf.pkl        – классификатор
    intent_v_word.pkl     – word-TF-IDF
    intent_v_char.pkl     – char-TF-IDF
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nltk.metrics.distance import edit_distance
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

from nlp_utils import clean_text, lemmatize_text


# ──────────────────────────────  CLASS  ──────────────────────────────
class IntentClassifier:
    """
    Word 1-2-gram + Char 3-5-gram  ➜  SGD(log_loss, early_stopping).
    """

    # ───────────── init ─────────────
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

        # — raw intents —
        with open(self.data_dir / "intents_dataset.json", encoding="utf-8") as f:
            raw = json.load(f)

        self.intents = {
            name: val
            for name, val in raw.items()
            if isinstance(val, dict) and "examples" in val
        }

        # — нормализованные примеры —
        self.norm_examples = {
            intent: [lemmatize_text(clean_text(ex)) for ex in obj["examples"]]
            for intent, obj in self.intents.items()
        }

        # — TF-IDF векторайзеры —
        self.v_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            tokenizer=str.split,
        )
        self.v_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,           # шумовые 3-граммы отсекаем
            sublinear_tf=True,
        )

        # — базовые гиперпараметры SGD —
        self._base_params = dict(
            loss="log_loss",          # чистая логистическая регрессия
            alpha=5e-5,
            learning_rate="optimal",
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42,
            early_stopping=False,
        )
        self.clf = self._new_clf()

    def _new_clf(self, **extra) -> SGDClassifier:
        """Cоздаёт новый SGDClassifier с нужными hyper-params."""
        params = self._base_params.copy()
        params.update(extra)
        return SGDClassifier(**params)

    # ───────────── public API ─────────────
    def train(
        self,
        *,
        plot: bool = True,
        epochs: int = 10,
        cv_folds: int = 5,
        test_size: float = 0.25,
        random_state: int = 42,
        elbow_max_k: int = 15,
        cm_block: int = 15,
    ) -> None:
        """
        epochs   – сколько раз вызвать partial_fit (и для history, и для финала)
        cm_block – сколько классов помещать на один confusion-PNG
        """
        # — подготовка датасета —
        X_raw, y = self._prepare_dataset()
        y = np.array(y, dtype=str)
        Xw = self.v_word.fit_transform(X_raw)
        Xc = self.v_char.fit_transform(X_raw)
        X_vec = sparse.hstack([Xw, Xc]).tocsr()

        # — балансировка классов —
        classes_ = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes_, y=y)
        cw = {c: w for c, w in zip(classes_, weights)}

        # — k-fold F1-оценка —
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        f1s = []
        for tr, vl in skf.split(X_vec, y):
            c = self._new_clf(class_weight=cw)
            c.fit(X_vec[tr], y[tr])
            f1s.append(
                f1_score(y[vl], c.predict(X_vec[vl]), average="macro", zero_division=0)
            )
        print(f"CV F1-macro: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

        # — графики —
        if plot:
            self._plot_training_history(
                clone(self._new_clf(class_weight=cw)),
                X_vec, y, epochs, test_size, random_state
            )
            self._plot_learning_curve(
                clone(self._new_clf(class_weight=cw)),
                X_vec, y, cv_folds
            )
            self._plot_confusion(
                clone(self._new_clf(class_weight=cw)),
                X_vec, y, test_size, random_state, block=cm_block
            )
            self._plot_elbow(X_vec, elbow_max_k)

        # — финальное обучение (partial_fit) —
        self.clf = self._new_clf(class_weight=cw)
        for _ in range(epochs):
            self.clf.partial_fit(X_vec, y, classes=classes_)

        # — сохраняем —
        pickle.dump(self.v_word, open("intent_v_word.pkl", "wb"))
        pickle.dump(self.v_char, open("intent_v_char.pkl", "wb"))
        pickle.dump(self.clf,    open("intent_clf.pkl",    "wb"))

    def load(self) -> None:
        """Загружает *.pkl, сохранённые train()."""
        self.v_word = pickle.load(open("intent_v_word.pkl", "rb"))
        self.v_char = pickle.load(open("intent_v_char.pkl", "rb"))
        self.clf    = pickle.load(open("intent_clf.pkl",    "rb"))

    # ───────────── инференс ─────────────
    def _vec(self, txt: str):
        Xw = self.v_word.transform([txt])
        Xc = self.v_char.transform([txt])
        return sparse.hstack([Xw, Xc])

    def predict(self, text: str) -> str:
        norm = lemmatize_text(clean_text(text))
        return self.clf.predict(self._vec(norm))[0]

    def predict_fuzzy(self, text: str, threshold: float = 0.25) -> str:
        """
        Если ближайший пример интента по Левенштейну далеко,
        ищет ближе среди всех интентов.
        """
        norm = lemmatize_text(clean_text(text))
        intent = self.predict(text)

        best = min(
            edit_distance(norm, ex) / max(1, len(ex))
            for ex in self.norm_examples[intent]
        )
        if best < threshold:
            return intent

        best_intent, best_d = None, threshold
        for cand, exs in self.norm_examples.items():
            for ex in exs:
                d = edit_distance(norm, ex) / max(1, len(ex))
                if d < best_d:
                    best_intent, best_d = cand, d
        return best_intent or intent

    # ───────────── helpers ─────────────
    def _prepare_dataset(self) -> Tuple[List[str], List[str]]:
        X_raw, y = [], []
        for intent, exs in self.norm_examples.items():
            X_raw.extend(exs)
            y.extend([intent] * len(exs))
        return X_raw, y

    # ─── 1. history (loss / acc) ───
    def _plot_training_history(
        self,
        clf: SGDClassifier,
        X_vec,
        y,
        epochs: int,
        test_size: float,
        random_state: int,
    ):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_vec, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        classes_ = np.unique(y)
        tr_loss, val_loss, val_acc = [], [], []

        for _ in range(epochs):
            clf.partial_fit(X_tr, y_tr, classes=classes_)

            p_tr  = clf.predict_proba(X_tr)
            p_val = clf.predict_proba(X_val)
            tr_loss.append(log_loss(y_tr,  p_tr,  labels=classes_))
            val_loss.append(log_loss(y_val, p_val, labels=classes_))
            val_acc.append(accuracy_score(y_val, clf.predict(X_val)))

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

        ax_l.plot(range(epochs), tr_loss,  label="Train Loss")
        ax_l.plot(range(epochs), val_loss, label="Validation Loss")
        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
        ax_l.set_title("Training and Validation Loss"); ax_l.legend()

        ax_r.plot(range(epochs), val_acc, color="green", label="Validation Accuracy")
        ax_r.set_xlabel("Epoch"); ax_r.set_ylabel("Accuracy")
        ax_r.set_title("Validation Accuracy"); ax_r.set_ylim(0, 1); ax_r.legend()

        plt.tight_layout(); plt.savefig("training_history.png", dpi=300); plt.close()

    # ─── 2. learning curve ───
    def _plot_learning_curve(self, clf, X_vec, y, cv_folds: int):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        tr_sizes, tr_scores, val_scores = learning_curve(
            clf, X_vec, y, cv=cv, n_jobs=-1
        )

        plt.figure(figsize=(8, 5))
        plt.plot(tr_sizes, tr_scores.mean(axis=1), marker="o", label="Train")
        plt.plot(tr_sizes, val_scores.mean(axis=1), marker="s", label="Validation")
        plt.xlabel("Training samples"); plt.ylabel("Accuracy")
        plt.title("Learning curve (SGD-LogReg)")
        plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
        plt.tight_layout(); plt.savefig("learning_curve.png", dpi=300); plt.close()

    # ─── 3. confusion matrix ───
    def _plot_confusion(
        self,
        clf,
        X_vec,
        y,
        test_size: float,
        random_state: int,
        *,
        block: int = 15,
        show_thr: float = 0.01,
    ):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_vec, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        cm = confusion_matrix(y_te, y_pred, labels=clf.classes_).astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm /= row_sums

        labels = np.array(clf.classes_)
        n_labels = len(labels)

        for start in range(0, n_labels, block):
            stop = min(start + block, n_labels)
            sub_lbl = labels[start:stop]
            cm_sub = cm[start:stop, start:stop]

            fig_size = max(8, 1 * len(sub_lbl))
            font_sz  = max(10, 220 // len(sub_lbl))

            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            im = ax.imshow(cm_sub, cmap="Blues", vmin=0, vmax=1)

            ax.set_xticks(np.arange(len(sub_lbl)), sub_lbl, rotation=90, fontsize=font_sz)
            ax.set_yticks(np.arange(len(sub_lbl)), sub_lbl, fontsize=font_sz)

            for i in range(cm_sub.shape[0]):
                for j in range(cm_sub.shape[1]):
                    v = cm_sub[i, j]
                    if v >= show_thr:
                        ax.text(
                            j, i, f"{v:.2f}",
                            ha="center", va="center",
                            color="white" if v > 0.5 else "black",
                            fontsize=font_sz,
                        )

            ax.set_xlabel("Predicted label", fontsize=font_sz)
            ax.set_ylabel("True label",      fontsize=font_sz)
            ax.set_title(f"Confusion matrix {start}-{stop-1}", fontsize=font_sz + 2)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=font_sz)

            plt.tight_layout()
            plt.savefig(f"confusion_matrix_{start}-{stop-1}.png", dpi=300)
            plt.close(fig)

    # ─── 4. elbow plot ───
    def _plot_elbow(self, X_vec, max_k: int = 15):
        inertias = []
        for k in range(1, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X_vec)
            inertias.append(km.inertia_)

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, max_k + 1), inertias, marker="o")
        plt.xlabel("K"); plt.ylabel("Inertia (SSE)")
        plt.title("Elbow plot for K-Means")
        plt.grid(alpha=0.5, linestyle="--")
        plt.tight_layout(); plt.savefig("elbow_curve.png", dpi=300); plt.close()


# ───────────── CLI ─────────────
if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    clf = IntentClassifier(DATA_DIR)
    clf.train(plot=True, epochs=10, elbow_max_k=15)
    print(
        "✓ Модель обучена и сохранена:\n"
        "   intent_clf.pkl / intent_v_word.pkl / intent_v_char.pkl\n"
        "✓ Графики: training_history.png, learning_curve.png, "
        "confusion_matrix_*.png, elbow_curve.png"
    )
