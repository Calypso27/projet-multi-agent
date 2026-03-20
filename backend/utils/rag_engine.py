"""Moteur RAG pour données tabulaires — recherche sémantique sur un DataFrame."""
import hashlib
import numpy as np
import pandas as pd
from typing import Optional

# Tentative d'import sentence-transformers (embeddings sémantiques)
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Fallback TF-IDF (sklearn — toujours disponible)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Constantes ───────────────────────────────────────────────────────────────
_MODEL_NAME   = "all-MiniLM-L6-v2"   # ~90 MB, rapide sur CPU
_MAX_ROWS     = 500                   # lignes max indexées
_ROWS_PER_CHUNK = 8                   # lignes par chunk


class DataRAG:
    """
    Moteur RAG pour interroger un DataFrame en langage naturel.

    Flux :
        rag = DataRAG()
        rag.index(df, filename)          # à faire une fois après le chargement
        context = rag.build_context(question, df)  # avant chaque appel LLM
    """

    def __init__(self):
        self._st_model: Optional[object] = None
        self._tfidf:    Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self.chunks:     list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.filename:   str = ""
        self.indexed:    bool = False
        self._dataset_hash: str = ""

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        """Hash MD5 du DataFrame pour détecter tout changement de contenu."""
        try:
            return hashlib.md5(
                pd.util.hash_pandas_object(df, index=True).values.tobytes()
            ).hexdigest()
        except Exception:
            return f"{df.shape[0]}_{df.shape[1]}"

    def needs_reindex(self, df: pd.DataFrame, filename: str) -> bool:
        """Retourne True si le dataset a changé depuis la dernière indexation."""
        if not self.indexed:
            return True
        if filename != self.filename:
            return True
        return self._hash_df(df) != self._dataset_hash

    # ── Modèle ────────────────────────────────────────────────────────────────

    def _get_st_model(self):
        if self._st_model is None:
            self._st_model = SentenceTransformer(_MODEL_NAME)
        return self._st_model

    # ── Construction des chunks ───────────────────────────────────────────────

    def _build_chunks(self, df: pd.DataFrame) -> list[str]:
        chunks = []

        # 1. Vue globale
        n_missing = df.isnull().sum().sum()
        pct_missing = n_missing / (df.shape[0] * df.shape[1]) * 100
        chunks.append(
            f"Vue globale du dataset '{self.filename}' : "
            f"{df.shape[0]:,} lignes × {df.shape[1]} colonnes. "
            f"Valeurs manquantes : {n_missing:,} ({pct_missing:.1f}%). "
            f"Colonnes : {', '.join(df.columns.tolist())}."
        )

        # 2. Description par colonne
        for col in df.columns:
            chunks.append(self._describe_column(df, col))

        # 3. Corrélations fortes (colonnes numériques)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    v = corr.iloc[i, j]
                    if abs(v) > 0.5:
                        pairs.append(f"{corr.columns[i]}↔{corr.columns[j]}: {v:.2f}")
            if pairs:
                chunks.append(
                    f"Corrélations notables (|r|>0.5) : {', '.join(pairs[:15])}."
                )

        # 4. Chunks de lignes (échantillon représentatif)
        sample = df.head(_MAX_ROWS)
        for start in range(0, len(sample), _ROWS_PER_CHUNK):
            batch = sample.iloc[start: start + _ROWS_PER_CHUNK]
            chunks.append(
                f"Lignes {start}–{start + len(batch) - 1} :\n"
                + batch.to_string(index=True)
            )

        return chunks

    @staticmethod
    def _describe_column(df: pd.DataFrame, col: str) -> str:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique  = s.nunique()

        if pd.api.types.is_numeric_dtype(s):
            return (
                f"Colonne numérique '{col}' : "
                f"min={s.min():.3g}, max={s.max():.3g}, "
                f"moyenne={s.mean():.3g}, écart-type={s.std():.3g}, "
                f"médiane={s.median():.3g}, "
                f"valeurs uniques={n_unique}, manquants={n_missing}."
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            return (
                f"Colonne date '{col}' : "
                f"de {s.min()} à {s.max()}, manquants={n_missing}."
            )
        else:
            top = s.value_counts().head(5).to_dict()
            return (
                f"Colonne catégorielle '{col}' : "
                f"{n_unique} valeurs uniques, "
                f"top 5 = {top}, manquants={n_missing}."
            )

    # ── Indexation ────────────────────────────────────────────────────────────

    def index(self, df: pd.DataFrame, filename: str = "dataset") -> None:
        """Indexe le DataFrame. Appelé une fois après le chargement."""
        self.filename      = filename
        self._dataset_hash = self._hash_df(df)
        self.chunks        = self._build_chunks(df)

        if _ST_AVAILABLE:
            model = self._get_st_model()
            self.embeddings = model.encode(
                self.chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            self._tfidf = None
        else:
            # Fallback TF-IDF
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            self._tfidf_matrix = self._tfidf.fit_transform(self.chunks)
            self.embeddings = None

        self.indexed = True

    # ── Recherche ─────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 6) -> list[str]:
        """Retourne les k chunks les plus pertinents pour la question."""
        if not self.indexed or not self.chunks:
            return []

        if _ST_AVAILABLE and self.embeddings is not None:
            model = self._get_st_model()
            q_emb = model.encode([query], normalize_embeddings=True)
            scores = (self.embeddings @ q_emb.T).flatten()
        else:
            q_vec = self._tfidf.transform([query])
            scores = cosine_similarity(self._tfidf_matrix, q_vec).flatten()

        top_k = int(min(k, len(self.chunks)))
        indices = np.argsort(scores)[-top_k:][::-1]
        return [self.chunks[i] for i in indices]

    # ── Construction du contexte pour le LLM ──────────────────────────────────

    def build_context(self, question: str, df: pd.DataFrame) -> str:
        """
        Construit le system prompt enrichi avec les chunks pertinents.
        Remplace summarize_df_for_llm() dans le chat.
        """
        if not self.indexed:
            # Fallback si pas encore indexé
            from backend.utils.llm_client import summarize_df_for_llm
            return (
                f"Tu es un assistant data science. Dataset : {self.filename}\n\n"
                f"{summarize_df_for_llm(df)}\n\n"
                f"Réponds en français. Sois précis et orienté action."
            )

        relevant = self.search(question, k=6)
        context_blocks = "\n\n---\n\n".join(relevant)

        method = "sentence-transformers" if _ST_AVAILABLE else "TF-IDF"
        return (
            f"Tu es un assistant data science expert.\n"
            f"Dataset analysé : **{self.filename}** "
            f"({df.shape[0]:,} lignes × {df.shape[1]} colonnes)\n\n"
            f"## Informations pertinentes (extraites par RAG/{method}) :\n\n"
            f"{context_blocks}\n\n"
            f"---\n"
            f"Réponds en français. Sois précis, factuel et orienté action. "
            f"Appuie-toi uniquement sur les données ci-dessus."
        )

    def is_ready(self) -> bool:
        return self.indexed

    def method(self) -> str:
        return "sentence-transformers" if _ST_AVAILABLE else "TF-IDF"


# ── Singleton partagé ─────────────────────────────────────────────────────────
_rag_instance: Optional[DataRAG] = None


def get_rag() -> DataRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = DataRAG()
    return _rag_instance
