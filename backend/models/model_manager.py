"""Gestionnaire de modèles ML - Sauvegarde et chargement"""
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List


class ModelManager:
    """
    Gère la sauvegarde, le chargement et la gestion des modèles ML entraînés

    Permet de:
    - Sauvegarder des modèles avec leurs métadonnées
    - Charger des modèles existants
    - Lister tous les modèles disponibles
    - Supprimer des modèles
    """

    MODELS_DIR = "saved_models"

    @staticmethod
    def _ensure_dir():
        """Crée le dossier de sauvegarde s'il n'existe pas"""
        if not os.path.exists(ModelManager.MODELS_DIR):
            os.makedirs(ModelManager.MODELS_DIR)
            print(f"[ModelManager] Dossier créé: {ModelManager.MODELS_DIR}")

    @staticmethod
    def save_model(model: Any, metadata: Dict[str, Any]) -> str:
        """
        Sauvegarde un modèle avec ses métadonnées

        Args:
            model: Le modèle scikit-learn entraîné
            metadata: Dictionnaire contenant:
                - model_name: Nom du modèle (ex: "Random Forest")
                - problem_type: "classification" ou "regression"
                - target_column: Nom de la colonne cible
                - feature_names: Liste des features utilisées
                - metrics: Dictionnaire des métriques de performance
                - preprocessing: Info sur le preprocessing appliqué

        Returns:
            model_id: Identifiant unique du modèle sauvegardé
        """
        ModelManager._ensure_dir()

        # Créer un ID unique basé sur timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name_clean = metadata.get('model_name', 'model').replace(' ', '_')
        model_id = f"{model_name_clean}_{timestamp}"

        # Chemins de sauvegarde
        model_path = os.path.join(ModelManager.MODELS_DIR, f"{model_id}_model.pkl")
        metadata_path = os.path.join(ModelManager.MODELS_DIR, f"{model_id}_metadata.json")

        # Sauvegarder le modèle
        joblib.dump(model, model_path)

        # Enrichir les métadonnées
        metadata_full = {
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
            'model_name': metadata.get('model_name'),
            'problem_type': metadata.get('problem_type'),
            'target_column': metadata.get('target_column'),
            'feature_names': metadata.get('feature_names', []),
            'metrics': metadata.get('metrics', {}),
            'preprocessing': metadata.get('preprocessing', {}),
            'model_path': model_path,
            'metadata_path': metadata_path
        }

        # Sauvegarder les métadonnées en JSON
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_full, f, indent=2, ensure_ascii=False)

        print(f"[ModelManager] Modèle sauvegardé: {model_id}")
        print(f"[ModelManager] - Type: {metadata_full['model_name']}")
        print(f"[ModelManager] - Problème: {metadata_full['problem_type']}")
        print(f"[ModelManager] - Métriques: {metadata_full['metrics']}")

        return model_id

    @staticmethod
    def load_model(model_id: str) -> tuple:
        """
        Charge un modèle sauvegardé

        Args:
            model_id: Identifiant du modèle

        Returns:
            (model, metadata): Tuple contenant le modèle et ses métadonnées
        """
        model_path = os.path.join(ModelManager.MODELS_DIR, f"{model_id}_model.pkl")
        metadata_path = os.path.join(ModelManager.MODELS_DIR, f"{model_id}_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_id}")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Métadonnées non trouvées: {model_id}")

        # Charger le modèle
        model = joblib.load(model_path)

        # Charger les métadonnées
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"[ModelManager] Modèle chargé: {model_id}")

        return model, metadata

    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        """
        Liste tous les modèles sauvegardés

        Returns:
            Liste de dictionnaires contenant les infos des modèles
        """
        ModelManager._ensure_dir()

        models = []

        for file in os.listdir(ModelManager.MODELS_DIR):
            if file.endswith("_metadata.json"):
                metadata_path = os.path.join(ModelManager.MODELS_DIR, file)

                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    models.append({
                        'id': metadata['model_id'],
                        'name': metadata['model_name'],
                        'type': metadata['problem_type'],
                        'target': metadata['target_column'],
                        'created_at': metadata['created_at'],
                        'metrics': metadata['metrics'],
                        'n_features': len(metadata.get('feature_names', []))
                    })
                except Exception as e:
                    print(f"[ModelManager] Erreur lecture {file}: {e}")

        # Trier par date de création (plus récent en premier)
        models.sort(key=lambda x: x['created_at'], reverse=True)

        return models

    @staticmethod
    def delete_model(model_id: str) -> bool:
        """
        Supprime un modèle sauvegardé

        Args:
            model_id: Identifiant du modèle

        Returns:
            True si suppression réussie, False sinon
        """
        model_path = os.path.join(ModelManager.MODELS_DIR, f"{model_id}_model.pkl")
        metadata_path = os.path.join(ModelManager.MODELS_DIR, f"{model_id}_metadata.json")

        success = True

        if os.path.exists(model_path):
            os.remove(model_path)
        else:
            success = False

        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        else:
            success = False

        if success:
            print(f"[ModelManager] Modèle supprimé: {model_id}")

        return success

    @staticmethod
    def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'un modèle sans le charger

        Args:
            model_id: Identifiant du modèle

        Returns:
            Dictionnaire des métadonnées ou None
        """
        metadata_path = os.path.join(ModelManager.MODELS_DIR, f"{model_id}_metadata.json")

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
