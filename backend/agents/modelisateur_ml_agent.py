"""Agent Modelisateur ML"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ..models.message import Message, MessageType
from .base_agent import BaseAgent
from ..models.model_manager import ModelManager
from ..utils.llm_client import call_llm, is_available

from collections import Counter

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelisateurMLAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ModelisateurML", role="Expert Machine Learning V2")
        self.trained_models = {}
        self.best_model = None
        self.feature_importance = {}

    def handle_message(self, message: Message):
        print(f"[ModelisateurML] Message reçu de {message.sender}, type: {message.message_type}")

        if not SKLEARN_AVAILABLE:
            print("[ModelisateurML] ERROR: scikit-learn non disponible")
            self._send_error(message.sender, "scikit-learn n'est pas installé")
            return

        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            print(f"[ModelisateurML] Tâche demandée: {task}")

            if task == "suggestion_modele":
                self._suggest_model(message)
            elif task == "entrainer":
                print("[ModelisateurML] Début de l'entraînement...")
                self._train_models(message)
            elif task == "hello":
                self._handle_hello(message)
        else:
            print(f"[ModelisateurML] Type de message non géré: {message.message_type}")

    def _suggest_model(self, message: Message):
        dataset = message.content.get("dataset")

        if dataset is None or dataset.empty:
            self._send_error(message.sender, "Aucun dataset fourni")
            return

        num_rows, num_cols = dataset.shape
        cat_cols = dataset.select_dtypes(include=['object', 'category']).columns
        
        suggestion = f"# Analyse du Dataset\n\n"
        suggestion += f"- Lignes : {num_rows}\n"
        suggestion += f"- Colonnes : {num_cols}\n"
        suggestion += f"- Colonnes catégorielles (Texte) détectées : {len(cat_cols)}\n\n"
        
        if len(cat_cols) > 0:
            suggestion += "**Note :** Je vais automatiquement convertir les colonnes texte en format numérique (One-Hot Encoding) pour l'entraînement.\n\n"
            
        suggestion += self._format_ml_suggestions(len(num_cols))
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.TASK_RESPONSE,
            content={'message': suggestion}
        ))

    def _train_models(self, message: Message):
        print(f"[ModelisateurML] _train_models appelée")
        dataset = message.content.get("dataset")
        target_column = message.content.get("target")
        problem_type = message.content.get("problem_type", "auto")

        print(f"[ModelisateurML] Dataset: {type(dataset)}, Target: {target_column}, Problem type: {problem_type}")

        if dataset is None or dataset.empty:
            print("[ModelisateurML] ERROR: Dataset vide ou None")
            self._send_error(message.sender, "Aucun dataset fourni")
            return

        print(f"[ModelisateurML] Dataset shape: {dataset.shape}")

        if not target_column:
            print("[ModelisateurML] ERROR: Target column manquante")
            self._send_error(message.sender, "Variable cible non spécifiée")
            return

        if target_column not in dataset.columns:
            print(f"[ModelisateurML] ERROR: Target '{target_column}' pas dans {list(dataset.columns)}")
            self._send_error(message.sender, f"Colonne '{target_column}' introuvable")
            return

        try:
            print("[ModelisateurML] Détection du type de problème...")
            if problem_type == "auto":
                problem_type = self._detect_problem_type(dataset, target_column)
            print(f"[ModelisateurML] Type de problème: {problem_type}")

            print("[ModelisateurML] Préparation des données...")
            X, y, feature_names = self._prepare_data_v2(dataset, target_column)
            print(f"[ModelisateurML] Données préparées: X.shape={X.shape}, y.shape={y.shape}, features={len(feature_names)}")

            if X.shape[1] == 0:
                print("[ModelisateurML] ERROR: Aucune feature après préparation")
                self._send_error(message.sender, "Aucune feature utilisable trouvée après nettoyage.")
                return

            print("[ModelisateurML] Split train/test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"[ModelisateurML] Train: {X_train.shape}, Test: {X_test.shape}")

            print("[ModelisateurML] Entraînement des modèles...")
            results = self._train_multiple_models_robust(
                X, y, X_train, X_test, y_train, y_test,
                problem_type, feature_names
            )
            print(f"[ModelisateurML] Entraînement terminé: {len(results)} modèles")

            if not results:
                print("[ModelisateurML] ERROR: Aucun modèle n'a réussi")
                self._send_error(message.sender, "Tous les modèles ont échoué lors de l'entraînement. Vérifiez la qualité des données.")
                return

            print("[ModelisateurML] Formatage des résultats...")
            response = self._format_training_results(results, problem_type, target_column)

            model_id = None
            if self.best_model is not None:
                try:
                    model_id = ModelManager.save_model(
                        model=self.best_model,
                        metadata={
                            'model_name': results[0]['name'],
                            'problem_type': problem_type,
                            'target_column': target_column,
                            'feature_names': feature_names,
                            'metrics': results[0]['metrics'],
                            'preprocessing': {
                                'one_hot_encoding': True,
                                'missing_value_strategy': 'median',
                                'high_cardinality_threshold': 20,
                                'test_size': 0.2,
                                'random_state': 42
                            }
                        }
                    )
                except Exception as e:
                    print(f"[ModelisateurML] ERREUR sauvegarde: {e}")

            print(f"[ModelisateurML] Envoi de la réponse à {message.sender}...")
            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={'message': response, 'results': results, 'model_id': model_id}
            ))
            print("[ModelisateurML] Réponse envoyée avec succès!")

        except Exception as e:
            import traceback
            print(f"Erreur critique ML Agent: {traceback.format_exc()}")
            self._send_error(message.sender, f"Erreur critique lors de l'entraînement: {str(e)}")

    def _detect_problem_type(self, df: pd.DataFrame, target: str) -> str:
        target_series = df[target]

        if pd.api.types.is_object_dtype(target_series) or pd.api.types.is_categorical_dtype(target_series):
            return "classification"

        if pd.api.types.is_numeric_dtype(target_series):
            unique_vals = target_series.nunique()
            if unique_vals < 10 or (unique_vals / len(target_series)) < 0.05:
                return "classification"
            return "regression"

        return "classification"

    def _prepare_data_v2(self, df: pd.DataFrame, target: str):
        X = df.drop(columns=[target])
        y = df[target]

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Supprimer les colonnes texte à haute cardinalité (identifiants, noms, etc.)
        # avant le one-hot encoding pour éviter l'explosion du nombre de features
        cat_cols = X.select_dtypes(include=['object']).columns
        high_card = [c for c in cat_cols if X[c].nunique() > 20]
        if high_card:
            print(f"[ModelisateurML] Colonnes haute cardinalité ignorées: {high_card}")
            X = X.drop(columns=high_card)

        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.median())

        mask = ~pd.Series(y).isna().values
        X = X[mask]
        y = np.array(y)[mask]

        feature_names = X.columns.tolist()
        return X.values, y, feature_names

    def _train_multiple_models_robust(self, X_full, y_full, X_train, X_test, y_train, y_test, problem_type, feature_names):
        results = []

        # ── Détection du déséquilibre de classes ──────────────────────────────
        is_imbalanced = False
        cv_scoring = 'r2' if problem_type == 'regression' else 'accuracy'

        if problem_type == 'classification':
            class_counts = Counter(y_full.tolist())
            min_ratio = min(class_counts.values()) / len(y_full)
            if min_ratio < 0.2:
                is_imbalanced = True
                cv_scoring = 'f1_weighted'
                print(f"[ModelisateurML] Déséquilibre détecté (classe min: {min_ratio:.1%}) → F1-weighted")

        # ── Cross-validation k=5 ──────────────────────────────────────────────
        n_splits = 5
        if problem_type == 'classification':
            n_classes = len(np.unique(y_full))
            n_splits = min(5, min(Counter(y_full.tolist()).values()))
            n_splits = max(2, n_splits)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        print(f"[ModelisateurML] Cross-validation {n_splits}-fold, métrique: {cv_scoring}")

        # ── Définition des modèles ─────────────────────────────────────────────
        if problem_type == "regression":
            models = {
                'Régression Linéaire': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
        else:
            models = {
                'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }

        best_score = -np.inf
        best_model_name = None

        for name, model in models.items():
            try:
                # 1. Score CV sur données complètes → critère de sélection
                cv_scores = cross_val_score(
                    model, X_full, y_full, cv=cv, scoring=cv_scoring, n_jobs=1
                )
                cv_mean = float(cv_scores.mean())
                cv_std  = float(cv_scores.std())
                print(f"[ModelisateurML] {name} — CV {cv_scoring}: {cv_mean:.4f} ±{cv_std:.4f}")

                # 2. Entraînement sur train set → métriques détaillées sur test set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == "regression":
                    r2   = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae  = mean_absolute_error(y_test, y_pred)
                    metrics = {'R²': round(r2, 4), 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}
                else:
                    accuracy = accuracy_score(y_test, y_pred)
                    metrics  = {'Accuracy': round(accuracy, 4)}
                    if len(np.unique(y_train)) == 2:
                        metrics['Precision'] = round(precision_score(y_test, y_pred, average='binary'), 4)
                        metrics['Recall']    = round(recall_score(y_test, y_pred, average='binary'), 4)
                        metrics['F1-Score']  = round(f1_score(y_test, y_pred, average='binary'), 4)

                importance = None
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    coefs = model.coef_
                    if len(coefs.shape) > 1:
                        coefs = coefs[0]
                    importance = dict(zip(feature_names, np.abs(coefs).flatten()))

                results.append({
                    'name': name,
                    'score': cv_mean,           # score CV = critère de classement
                    'cv_std': round(cv_std, 4),
                    'cv_metric': cv_scoring,
                    'metrics': metrics,
                    'feature_importance': importance,
                    'is_imbalanced': is_imbalanced,
                    'error': None
                })

                if cv_mean > best_score:
                    best_score     = cv_mean
                    best_model_name = name
                    self.best_model = model
                    self.feature_importance = importance

            except Exception as e:
                results.append({
                    'name': name,
                    'score': -1,
                    'cv_std': None,
                    'cv_metric': cv_scoring,
                    'metrics': {},
                    'feature_importance': None,
                    'is_imbalanced': is_imbalanced,
                    'error': str(e)
                })

        successful_results = [r for r in results if r['error'] is None]
        successful_results.sort(key=lambda x: x['score'], reverse=True)

        for r in successful_results:
            r['is_best'] = (r['name'] == best_model_name)

        return successful_results

    def _format_training_results(self, results, problem_type, target):
        output = f"# Résultats de l'Entraînement\n\n"
        output += f"**Variable cible:** {target}\n"
        output += f"**Type:** {'Régression' if problem_type == 'regression' else 'Classification'}\n"

        # Infos sur la stratégie d'évaluation
        if results:
            cv_metric = results[0].get('cv_metric', '')
            is_imbalanced = results[0].get('is_imbalanced', False)
            if is_imbalanced:
                output += f"**Données déséquilibrées détectées** → sélection par **F1-weighted** (cross-validation 5-fold)\n\n"
            else:
                label = 'R² (CV 5-fold)' if problem_type == 'regression' else 'Accuracy (CV 5-fold)'
                output += f"**Critère de sélection :** {label}\n\n"

        output += "## Comparaison des Modèles\n\n"

        if problem_type == "regression":
            output += "| Modèle | CV R² moyen | ±Écart-type | R² test | RMSE | MAE | Statut |\n"
            output += "|--------|-------------|-------------|---------|------|-----|--------|\n"
            for r in results:
                status = "✓ Meilleur" if r.get('is_best') else "✓"
                if r['score'] == -1:
                    status = "✗ Erreur"
                cv_mean = f"{r['score']:.4f}" if r['score'] != -1 else '-'
                cv_std  = f"{r['cv_std']:.4f}" if r.get('cv_std') is not None else '-'
                output += (f"| {r['name']} | {cv_mean} | {cv_std} | "
                           f"{r['metrics'].get('R²', '-')} | {r['metrics'].get('RMSE', '-')} | "
                           f"{r['metrics'].get('MAE', '-')} | {status} |\n")
        else:
            is_imbalanced = results[0].get('is_imbalanced', False) if results else False
            cv_label = "CV F1-weighted" if is_imbalanced else "CV Accuracy"
            output += f"| Modèle | {cv_label} moyen | ±Écart-type | Accuracy test | Precision | Recall | F1-Score | Statut |\n"
            output += "|--------|----------------|-------------|---------------|-----------|--------|----------|--------|\n"
            for r in results:
                status = "✓ Meilleur" if r.get('is_best') else "✓"
                if r['score'] == -1:
                    status = "✗ Erreur"
                cv_mean = f"{r['score']:.4f}" if r['score'] != -1 else '-'
                cv_std  = f"{r['cv_std']:.4f}" if r.get('cv_std') is not None else '-'
                output += (f"| {r['name']} | {cv_mean} | {cv_std} | "
                           f"{r['metrics'].get('Accuracy', '-')} | {r['metrics'].get('Precision', '-')} | "
                           f"{r['metrics'].get('Recall', '-')} | {r['metrics'].get('F1-Score', '-')} | {status} |\n")

        if results and results[0].get('is_best'):
            best = results[0]
            output += f"\n## Meilleur Modèle: {best['name']}\n\n"
            output += f"**Score CV moyen :** {best['score']:.4f} ±{best.get('cv_std', 0):.4f}\n\n"

            if problem_type == "regression":
                output += f"Le modèle peut expliquer **{best['metrics']['R²']*100:.1f}%** de la variance sur le jeu de test.\n"
            else:
                output += f"Précision de classification (test) : **{best['metrics']['Accuracy']*100:.1f}%**\n"

            if best['feature_importance']:
                output += "\n### Top 5 Variables Importantes\n\n"
                sorted_feats = sorted(best['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                for f, val in sorted_feats:
                    output += f"- **{f}**: {val:.4f}\n"

            # === LLM : explication du modèle en langage naturel ===
            if is_available():
                try:
                    top_features = []
                    if best['feature_importance']:
                        top_features = sorted(
                            best['feature_importance'].items(),
                            key=lambda x: x[1], reverse=True
                        )[:5]

                    prompt = (
                        f"Un modèle de Machine Learning vient d'être entraîné :\n"
                        f"- Type de problème : {problem_type}\n"
                        f"- Variable à prédire : {target}\n"
                        f"- Meilleur algorithme : {best['name']}\n"
                        f"- Métriques : {best['metrics']}\n"
                        f"- Variables les plus importantes : {top_features}\n\n"
                        f"Explique ces résultats en 4-5 phrases accessibles à un chef de projet "
                        f"(non-expert ML). Dis si le modèle est bon ou non, pourquoi, "
                        f"et ce que les variables importantes signifient concrètement."
                    )
                    explanation = call_llm(
                        prompt=prompt,
                        system="Tu es un expert ML pédagogue. Traduis les métriques en langage métier clair.",
                    )
                    if explanation:
                        output += f"\n### Interprétation par l'IA\n\n{explanation}\n"
                except Exception:
                    pass  # LLM optionnel
            # =======================================================

        return output

    def _format_ml_suggestions(self, num_cols):
        return "Utilisez la tâche 'entrainer' avec le dataset pour lancer l'analyse."

    def _handle_hello(self, message: Message):
        status = "disponible" if SKLEARN_AVAILABLE else "indisponible"
        self.message_bus.send_message(Message(
            sender=self.name, receiver=message.sender,
            message_type=MessageType.TASK_RESPONSE,
            content={'message': f"Agent Data Scientist V2 prêt. Scikit-learn: {status}"}
        ))
    
    def _send_error(self, recipient: str, error_message: str):
        self.message_bus.send_message(Message(
            sender=self.name, receiver=recipient,
            message_type=MessageType.ERROR,
            content={'error': error_message}
        ))