"""
Évaluateur RAG — métriques sans appel LLM.

Métriques de retrieval :
  - retrieval_score   : similarité cosinus moyenne top-k (0→1)
  - score_gap         : discrimination (max - min des scores)
  - keyword_coverage  : % mots de la question trouvés dans les chunks (proxy Recall)
  - precision_at_k    : % chunks contenant ≥1 expected_keyword, mot entier (proxy Precision@k)
  - success_rate      : 1 si ≥1 chunk pertinent, 0 sinon
  - mrr               : Mean Reciprocal Rank (rang 1er chunk pertinent)

Métriques post-génération (nécessitent la réponse LLM) :
  - faithfulness      : % mots de la réponse ancrés dans le contexte (proxy anti-hallucination)
  - answer_relevance  : chevauchement vocabulaire question ↔ réponse
"""
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from .rag_engine import DataRAG


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    query:            str
    chunks:           list[str]
    scores:           list[float]
    # Métriques cosinus
    retrieval_score:  float   # similarité moyenne top-k
    score_gap:        float   # discrimination
    keyword_coverage: float   # proxy Recall (mots de la question dans les chunks)
    # Métriques standards (nécessitent expected_keywords)
    precision_at_k:   float   # proxy Precision@k
    success_rate:     float   # 1.0 si succès, 0.0 sinon
    mrr:              float   # Mean Reciprocal Rank
    verdict:          str     # "Excellent" / "Bon" / "Moyen" / "Faible"
    # Métriques post-génération (optionnelles)
    faithfulness:     float = 0.0  # % mots réponse LLM fondés sur le contexte
    answer_relevance: float = 0.0  # chevauchement vocabulaire question ↔ réponse


@dataclass
class BenchmarkQuestion:
    query:             str
    expected_keywords: list[str]   # mots qui DOIVENT apparaître dans les chunks
    category:          str         # "schema" | "stats" | "qualite" | "correlations"


@dataclass
class BenchmarkReport:
    questions:        list[BenchmarkQuestion]
    results:          list[RetrievalResult]
    # Métriques globales
    overall_score:    float   # moyenne retrieval_score
    coverage_score:   float   # moyenne keyword_coverage
    mean_precision:   float   # moyenne Precision@k
    mean_success_rate: float  # % questions avec au moins 1 chunk pertinent
    mean_mrr:         float   # moyenne MRR
    per_category:     dict    # scores par catégorie
    config:           dict    # paramètres RAG utilisés


# ─── Évaluateur ───────────────────────────────────────────────────────────────

class RAGEvaluator:
    """Évalue la qualité du retrieval sans appel LLM."""

    RELEVANCE_THRESHOLD = 0.25  # score cosinus minimum pour considérer un chunk pertinent

    _STOPWORDS = {
        "le","la","les","de","du","des","un","une","et","est","en","dans","pour",
        "que","qui","quelles","quels","quel","quelle","avec","sur","par","il",
        "elle","sont","ont","a","au","aux","ce","cette","ces","y","combien",
        "comment","plus","moins","très","bien","pas","ne","se","si","ou","mais"
    }

    # ── Méthode utilitaire : matching mot entier ──────────────────────────────

    @staticmethod
    def _contains_keyword(keyword: str, text: str) -> bool:
        """Vérifie la présence du mot-clé comme mot entier (pas substring partielle)."""
        try:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            return bool(re.search(pattern, text.lower()))
        except re.error:
            return keyword.lower() in text.lower()

    # ── Évaluation d'une seule requête ────────────────────────────────────────

    def evaluate_query(
        self,
        rag: DataRAG,
        query: str,
        k: int = 6,
        expected_keywords: Optional[list[str]] = None
    ) -> RetrievalResult:
        """Évalue la qualité du retrieval pour une question donnée."""
        if not rag.is_ready():
            return self._empty_result(query, "Non indexé")

        chunks, scores = self._search_with_scores(rag, query, k)
        if not scores:
            return self._empty_result(query, "Aucun résultat")

        retrieval_score  = float(np.mean(scores))
        score_gap        = float(max(scores) - min(scores))
        keyword_coverage = self._keyword_coverage(query, chunks)

        # Métriques standards basées sur expected_keywords
        if expected_keywords:
            precision_at_k = self._precision_at_k(expected_keywords, chunks)
            success_rate   = 1.0 if precision_at_k > 0 else 0.0
            mrr            = self._mrr(expected_keywords, chunks)
        else:
            # Sans vérité terrain : estimation par seuil cosinus
            n_relevant     = sum(1 for s in scores if s >= self.RELEVANCE_THRESHOLD)
            precision_at_k = n_relevant / len(scores)
            success_rate   = 1.0 if n_relevant > 0 else 0.0
            mrr            = self._mrr_by_score(scores)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            scores=[round(s, 4) for s in scores],
            retrieval_score=round(retrieval_score, 4),
            score_gap=round(score_gap, 4),
            keyword_coverage=round(keyword_coverage, 4),
            precision_at_k=round(precision_at_k, 4),
            success_rate=round(success_rate, 4),
            mrr=round(mrr, 4),
            verdict=self._verdict(retrieval_score, precision_at_k)
        )

    # ── Benchmark auto-généré ─────────────────────────────────────────────────

    def generate_benchmark(self, df: pd.DataFrame) -> list[BenchmarkQuestion]:
        """Génère automatiquement un jeu de questions à partir du DataFrame."""
        questions = []
        cols     = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # ── Schéma ──
        questions.append(BenchmarkQuestion(
            query="Quelles sont les colonnes du dataset ?",
            expected_keywords=cols[:5],
            category="schema"
        ))
        questions.append(BenchmarkQuestion(
            query="Combien de lignes et colonnes dans le dataset ?",
            expected_keywords=[str(df.shape[0]), str(df.shape[1])],
            category="schema"
        ))

        # ── Stats numériques ──
        for col in num_cols[:3]:
            mean_val = f"{df[col].mean():.2f}"
            questions.append(BenchmarkQuestion(
                query=f"Quelle est la moyenne de la colonne {col} ?",
                expected_keywords=[col, mean_val[:4]],
                category="stats"
            ))
            questions.append(BenchmarkQuestion(
                query=f"Quelles sont les statistiques de {col} ?",
                expected_keywords=[col, "min", "max"],
                category="stats"
            ))

        # ── Qualité ──
        n_missing = int(df.isnull().sum().sum())
        questions.append(BenchmarkQuestion(
            query="Y a-t-il des valeurs manquantes dans le dataset ?",
            expected_keywords=["manquants", str(n_missing)],
            category="qualite"
        ))
        missing_cols = df.columns[df.isnull().sum() > 0].tolist()
        if missing_cols:
            questions.append(BenchmarkQuestion(
                query="Quelles colonnes ont des valeurs manquantes ?",
                expected_keywords=missing_cols[:3],
                category="qualite"
            ))

        # ── Catégorielles ──
        for col in cat_cols[:2]:
            if len(df[col].dropna()) > 0:
                top_val = str(df[col].value_counts().index[0])
                questions.append(BenchmarkQuestion(
                    query=f"Quelles sont les valeurs les plus fréquentes de {col} ?",
                    expected_keywords=[col, top_val],
                    category="schema"
                ))

        # ── Corrélations ──
        if len(num_cols) >= 2:
            corr     = df[num_cols].corr()
            best_pair, best_val = None, 0.0
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    v = abs(corr.iloc[i, j])
                    if v > best_val:
                        best_val  = v
                        best_pair = (corr.columns[i], corr.columns[j])
            if best_pair and best_val > 0.3:
                questions.append(BenchmarkQuestion(
                    query=f"Y a-t-il une corrélation entre {best_pair[0]} et {best_pair[1]} ?",
                    expected_keywords=[best_pair[0], best_pair[1], "corrélation"],
                    category="correlations"
                ))

        return questions

    def run_benchmark(self, rag: DataRAG, df: pd.DataFrame, k: int = 6) -> BenchmarkReport:
        """Exécute le benchmark complet et retourne un rapport."""
        questions = self.generate_benchmark(df)
        results   = []

        for q in questions:
            result = self.evaluate_query(rag, q.query, k=k,
                                         expected_keywords=q.expected_keywords)
            results.append(result)

        # Métriques globales
        overall_score     = float(np.mean([r.retrieval_score  for r in results])) if results else 0.0
        coverage_score    = float(np.mean([r.keyword_coverage for r in results])) if results else 0.0
        mean_precision    = float(np.mean([r.precision_at_k   for r in results])) if results else 0.0
        mean_success_rate = float(np.mean([r.success_rate     for r in results])) if results else 0.0
        mean_mrr          = float(np.mean([r.mrr              for r in results])) if results else 0.0

        # Scores par catégorie (retrieval_score + precision_at_k)
        per_category: dict = {}
        for q, r in zip(questions, results):
            per_category.setdefault(q.category, {"retrieval": [], "precision": [], "mrr": []})
            per_category[q.category]["retrieval"].append(r.retrieval_score)
            per_category[q.category]["precision"].append(r.precision_at_k)
            per_category[q.category]["mrr"].append(r.mrr)
        per_category = {
            cat: {
                "retrieval":  round(float(np.mean(v["retrieval"])),  4),
                "precision":  round(float(np.mean(v["precision"])),  4),
                "mrr":        round(float(np.mean(v["mrr"])),        4),
            }
            for cat, v in per_category.items()
        }

        return BenchmarkReport(
            questions=questions,
            results=results,
            overall_score=round(overall_score, 4),
            coverage_score=round(coverage_score, 4),
            mean_precision=round(mean_precision, 4),
            mean_success_rate=round(mean_success_rate, 4),
            mean_mrr=round(mean_mrr, 4),
            per_category=per_category,
            config={
                "method":   rag.method(),
                "k":        k,
                "n_chunks": len(rag.chunks),
                "model":    "all-MiniLM-L6-v2" if rag.method() == "sentence-transformers" else "TF-IDF",
            }
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(query: str, verdict: str) -> RetrievalResult:
        return RetrievalResult(
            query=query, chunks=[], scores=[],
            retrieval_score=0.0, score_gap=0.0, keyword_coverage=0.0,
            precision_at_k=0.0, success_rate=0.0, mrr=0.0,
            verdict=verdict
        )

    @staticmethod
    def _search_with_scores(rag: DataRAG, query: str, k: int):
        """Retourne (chunks, scores) pour une requête."""
        if not rag.indexed or not rag.chunks:
            return [], []
        if rag.embeddings is not None:
            try:
                model  = rag._get_st_model()
                q_emb  = model.encode([query], normalize_embeddings=True)
                scores = (rag.embeddings @ q_emb.T).flatten()
            except Exception:
                return [], []
        else:
            if rag._tfidf is None:
                return [], []
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec  = rag._tfidf.transform([query])
            scores = cosine_similarity(rag._tfidf_matrix, q_vec).flatten()

        top_k   = min(k, len(rag.chunks))
        indices = np.argsort(scores)[-top_k:][::-1]
        return [rag.chunks[i] for i in indices], [float(scores[i]) for i in indices]

    def _precision_at_k(self, expected_keywords: list[str], chunks: list[str]) -> float:
        """Proxy Precision@k : % des chunks retournés contenant ≥1 expected_keyword (mot entier)."""
        if not chunks or not expected_keywords:
            return 0.0
        relevant = sum(
            1 for chunk in chunks
            if any(self._contains_keyword(kw, chunk) for kw in expected_keywords)
        )
        return relevant / len(chunks)

    def _mrr(self, expected_keywords: list[str], chunks: list[str]) -> float:
        """
        Mean Reciprocal Rank : 1/rang du 1er chunk contenant ≥1 expected_keyword (mot entier).
        0.0 si aucun chunk ne contient les mots-clés.
        """
        if not chunks or not expected_keywords:
            return 0.0
        for rank, chunk in enumerate(chunks, start=1):
            if any(self._contains_keyword(kw, chunk) for kw in expected_keywords):
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _mrr_by_score(scores: list[float]) -> float:
        """Fallback MRR sans expected_keywords : 1/rang du 1er chunk au-dessus du seuil."""
        threshold = 0.25
        for rank, score in enumerate(scores, start=1):
            if score >= threshold:
                return 1.0 / rank
        return 0.0

    def _keyword_coverage(self, query: str, chunks: list[str]) -> float:
        """Proxy Recall : % des mots significatifs de la question présents dans les chunks (mot entier)."""
        words = [w.lower().strip("?!.,") for w in query.split()
                 if w.lower() not in self._STOPWORDS and len(w) > 2]
        if not words:
            return 1.0
        combined = " ".join(chunks)
        return sum(1 for w in words if self._contains_keyword(w, combined)) / len(words)

    def evaluate_faithfulness(self, llm_response: str, context_chunks: list[str]) -> float:
        """
        Proxy Faithfulness : % des mots significatifs de la réponse LLM
        présents dans le contexte RAG.
        Mesure : la réponse s'appuie-t-elle sur les données fournies (pas d'hallucination) ?
        """
        if not llm_response or not context_chunks:
            return 0.0
        combined_context = " ".join(context_chunks)
        response_words = [
            w.lower().strip(".,;:!?\"'()[]") for w in llm_response.split()
            if w.lower() not in self._STOPWORDS and len(w) > 2
        ]
        if not response_words:
            return 1.0
        found = sum(1 for w in response_words if self._contains_keyword(w, combined_context))
        return round(found / len(response_words), 4)

    def evaluate_answer_relevance(self, question: str, llm_response: str) -> float:
        """
        Proxy Answer Relevance : chevauchement de vocabulaire entre question et réponse.
        Mesure : la réponse adresse-t-elle vraiment la question posée ?
        """
        if not question or not llm_response:
            return 0.0
        q_words = {w.lower().strip("?!.,") for w in question.split()
                   if w.lower() not in self._STOPWORDS and len(w) > 2}
        r_words = {w.lower().strip(".,;:!?") for w in llm_response.split()
                   if w.lower() not in self._STOPWORDS and len(w) > 2}
        if not q_words:
            return 0.0
        return round(len(q_words & r_words) / len(q_words), 4)

    @staticmethod
    def _verdict(retrieval_score: float, precision_at_k: float) -> str:
        combined = (retrieval_score + precision_at_k) / 2
        if combined >= 0.55:
            return "Excellent"
        if combined >= 0.35:
            return "Bon"
        if combined >= 0.20:
            return "Moyen"
        return "Faible"
