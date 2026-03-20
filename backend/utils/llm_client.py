"""Client LLM partagé — Anthropic Claude ou Ollama (local)"""
import os
import json
import pandas as pd
from pathlib import Path

# Charge .env depuis la racine du projet (deux niveaux au-dessus de ce fichier)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# ── Anthropic ─────────────────────────────────────────────────────────────────
try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

_anthropic_client = None

# ── Ollama ────────────────────────────────────────────────────────────────────
_OLLAMA_BASE_URL      = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_DEFAULT_MODEL = "llama3.2:latest"   # modèle utilisé par défaut


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


# ── Disponibilité ─────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Vrai si Anthropic API ou Ollama est disponible."""
    return _anthropic_available() or _ollama_available()


def _anthropic_available() -> bool:
    return _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))


def _ollama_available() -> bool:
    """Vérifie si le serveur Ollama répond."""
    try:
        import requests
        r = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_backend() -> str:
    """Retourne 'anthropic', 'ollama' ou 'none'."""
    if _anthropic_available():
        return "anthropic"
    if _ollama_available():
        return "ollama"
    return "none"


def list_ollama_models() -> list[str]:
    """Retourne la liste des modèles Ollama installés."""
    try:
        import requests
        r = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


# ── Appel LLM unifié ──────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    system: str = "",
    model: str = "claude-haiku-4-5",
) -> str:
    """
    Appelle le LLM disponible (Anthropic en priorité, sinon Ollama).
    Retourne "" en cas d'erreur ou d'indisponibilité.
    """
    backend = get_backend()

    if backend == "anthropic":
        return _call_anthropic(prompt, system, model)

    if backend == "ollama":
        ollama_model = os.environ.get("OLLAMA_MODEL", _OLLAMA_DEFAULT_MODEL)
        return _call_ollama(prompt, system, ollama_model)

    return ""


def _call_anthropic(prompt: str, system: str, model: str) -> str:
    """Appel via l'API Anthropic."""
    try:
        client = _get_anthropic_client()
        kwargs = {
            "model":      model,
            "max_tokens": 1024,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()
    except Exception:
        return ""


def _call_ollama(prompt: str, system: str, model: str) -> str:
    """Appel via l'API REST Ollama (localhost)."""
    try:
        import requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = requests.post(
            f"{_OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["message"]["content"].strip()
    except Exception:
        pass
    return ""


# ── Résumé DataFrame ──────────────────────────────────────────────────────────

def summarize_df_for_llm(df: pd.DataFrame, max_sample_rows: int = 5) -> str:
    """
    Résume un DataFrame en texte compact pour l'envoyer à un LLM
    sans dépasser les limites de tokens.
    """
    if df is None or df.empty:
        return "Dataset vide."

    lines = [
        f"Dimensions: {df.shape[0]:,} lignes × {df.shape[1]} colonnes",
        f"Colonnes: {df.columns.tolist()}",
        f"Types: {df.dtypes.to_dict()}",
    ]

    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_info = {col: int(cnt) for col, cnt in missing[missing > 0].items()}
        lines.append(f"Valeurs manquantes: {missing_info}")
    else:
        lines.append("Valeurs manquantes: aucune")

    lines.append(f"Lignes dupliquées: {df.duplicated().sum()}")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe().round(2).to_dict()
        lines.append(f"Statistiques numériques: {desc}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        card = {col: df[col].nunique() for col in cat_cols}
        lines.append(f"Cardinalité catégorielle: {card}")

    if len(numeric_cols) >= 2:
        corr   = df[numeric_cols].corr()
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    strong.append(f"{corr.columns[i]}↔{corr.columns[j]}: {val:.2f}")
        if strong:
            lines.append(f"Corrélations fortes (|r|>0.7): {strong[:10]}")

    sample = df.head(max_sample_rows).to_dict(orient="records")
    lines.append(f"Échantillon ({max_sample_rows} lignes): {sample}")

    return "\n".join(lines)
