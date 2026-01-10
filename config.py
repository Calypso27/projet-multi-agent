"""
Configuration globale du projet
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger variables d'environnement
load_dotenv()

# Chemins
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Créer les dossiers s'ils n'existent pas
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuration OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Paramètres des agents
AGENT_CONFIG = {
    "chef_projet": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 500
    },
    "ingenieur_data": {
        "max_file_size_mb": 100,
        "supported_formats": [".csv", ".xlsx", ".xls"]
    },
    "analyste": {
        "outlier_threshold": 3.0,
        "correlation_threshold": 0.5,
        "max_plots": 10
    },
    "modelisateur": {
        "test_size": 0.2,
        "random_state": 42,
        "max_training_time": 300,  # secondes
        "cv_folds": 5
    }
}

# Debug mode
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Logging
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"

print(f" Configuration chargée - Environnement: {ENVIRONMENT}")