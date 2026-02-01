"""Agent Modelisateur ML"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ..models.message import Message, MessageType
from .base_agent import BaseAgent
from ..models.model_manager import ModelManager

try:
    from sklearn.model_selection import train_test_split
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
                X_train, X_test, y_train, y_test,
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
                print("[ModelisateurML] Sauvegarde du meilleur modèle...")
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
                                'test_size': 0.2,
                                'random_state': 42
                            }
                        }
                    )
                    print(f"[ModelisateurML] Modèle sauvegardé avec succès: {model_id}")
                except Exception as e:
                    print(f"[ModelisateurML] ERREUR lors de la sauvegarde du modèle: {str(e)}")
            else:
                print("[ModelisateurML] Aucun modèle à sauvegarder (best_model est None)")

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

        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.median())

        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        feature_names = X.columns.tolist()
        return X.values, y.values, feature_names

    def _train_multiple_models_robust(self, X_train, X_test, y_train, y_test, problem_type, feature_names):
        results = []

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
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == "regression":
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    score = r2
                    metrics = {'R²': round(r2, 4), 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}
                else:
                    accuracy = accuracy_score(y_test, y_pred)
                    score = accuracy
                    metrics = {'Accuracy': round(accuracy, 4)}
                    
                    if len(np.unique(y_train)) == 2:
                        metrics['Precision'] = round(precision_score(y_test, y_pred, average='binary'), 4)
                        metrics['Recall'] = round(recall_score(y_test, y_pred, average='binary'), 4)
                        metrics['F1-Score'] = round(f1_score(y_test, y_pred, average='binary'), 4)

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
                    'score': score,
                    'metrics': metrics,
                    'feature_importance': importance,
                    'error': None
                })

                if score > best_score:
                    best_score = score
                    best_model_name = name
                    self.best_model = model
                    self.feature_importance = importance

            except Exception as e:
                results.append({
                    'name': name,
                    'score': -1,
                    'metrics': {},
                    'feature_importance': None,
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
        output += f"**Type:** {'Régression' if problem_type == 'regression' else 'Classification'}\n\n"
        
        output += "## Comparaison des Modèles\n\n"
        
        if problem_type == "regression":
            output += "| Modèle | R² | RMSE | MAE | Statut |\n"
            output += "|--------|-----|------|-----|--------|\n"
            for r in results:
                status = "✓ Meilleur" if r.get('is_best') else "✓"
                if r['score'] == -1: status = "✗ Erreur"
                output += f"| {r['name']} | {r['metrics'].get('R²', '-')} | {r['metrics'].get('RMSE', '-')} | {r['metrics'].get('MAE', '-')} | {status} |\n"
        else:
            output += "| Modèle | Accuracy | Precision | Recall | F1-Score | Statut |\n"
            output += "|--------|----------|-----------|--------|----------|--------|\n"
            for r in results:
                status = "✓ Meilleur" if r.get('is_best') else "✓"
                if r['score'] == -1: status = "✗ Erreur"

                prec = r['metrics'].get('Precision', '-')
                rec = r['metrics'].get('Recall', '-')
                f1 = r['metrics'].get('F1-Score', '-')
                acc = r['metrics'].get('Accuracy', '-')
                
                output += f"| {r['name']} | {acc} | {prec} | {rec} | {f1} | {status} |\n"

        if results and results[0].get('is_best'):
            best = results[0]
            output += f"\n## Meilleur Modèle: {best['name']}\n\n"

            if problem_type == "regression":
                output += f"Le modèle peut expliquer **{best['metrics']['R²']*100:.1f}%** de la variance.\n"
            else:
                output += f"Précision de classification: **{best['metrics']['Accuracy']*100:.1f}%**\n"

            if best['feature_importance']:
                output += "\n### Top 5 Variables Importantes\n\n"
                sorted_feats = sorted(best['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
                for f, val in sorted_feats:
                    output += f"- **{f}**: {val:.4f}\n"

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