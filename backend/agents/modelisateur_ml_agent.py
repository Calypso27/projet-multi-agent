"""Agent Modélisateur ML - Entraînement automatisé"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ..models.message import Message, MessageType
from .base_agent import BaseAgent

# ML imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelisateurMLAgent(BaseAgent):
    """
    Agent Modélisateur ML
    - Entraînement automatisé
    - Régression et Classification
    - Sélection automatique du meilleur modèle
    """
    
    def __init__(self):
        super().__init__(name="ModelisateurML", role="Expert Machine Learning")
        self.trained_models = {}
        self.best_model = None
        self.feature_importance = {}
    
    def handle_message(self, message: Message):
        """Traite les messages"""
        
        if not SKLEARN_AVAILABLE:
            self._send_error(message.sender, "scikit-learn n'est pas installé")
            return
        
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            
            if task == "suggestion_modele":
                self._suggest_model(message)
            elif task == "entrainer":
                self._train_models(message)
            elif task == "hello":
                self._handle_hello(message)
    
    def _suggest_model(self, message: Message):
        """Suggère des modèles appropriés (version simple)"""
        dataset = message.content.get("dataset")
        
        if dataset is None or dataset.empty:
            self._send_error(message.sender, "Aucun dataset fourni")
            return
        
        numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
        
        suggestion = self._format_ml_suggestions(len(numeric_cols))
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.TASK_RESPONSE,
            content={'message': suggestion}
        ))
    
    def _train_models(self, message: Message):
        """Entraîne automatiquement plusieurs modèles"""
        dataset = message.content.get("dataset")
        target_column = message.content.get("target")
        problem_type = message.content.get("problem_type", "auto")
        
        if dataset is None or dataset.empty:
            self._send_error(message.sender, "Aucun dataset fourni")
            return
        
        if not target_column:
            self._send_error(message.sender, "Variable cible non spécifiée")
            return
        
        if target_column not in dataset.columns:
            self._send_error(message.sender, f"Colonne '{target_column}' introuvable")
            return
        
        try:
            # Déterminer le type de problème
            if problem_type == "auto":
                problem_type = self._detect_problem_type(dataset, target_column)
            
            # Préparer les données
            X, y, feature_names = self._prepare_data(dataset, target_column)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Entraîner plusieurs modèles
            results = self._train_multiple_models(
                X_train, X_test, y_train, y_test, 
                problem_type, feature_names
            )
            
            # Formater les résultats
            response = self._format_training_results(results, problem_type, target_column)
            
            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={'message': response, 'results': results}
            ))
            
        except Exception as e:
            self._send_error(message.sender, f"Erreur lors de l'entraînement: {str(e)}")
    
    def _detect_problem_type(self, df: pd.DataFrame, target: str) -> str:
        """Détecte automatiquement s'il s'agit de régression ou classification"""
        target_series = df[target]
        
        # Si numérique et beaucoup de valeurs uniques → régression
        if pd.api.types.is_numeric_dtype(target_series):
            unique_ratio = target_series.nunique() / len(target_series)
            if unique_ratio > 0.05:  # Plus de 5% de valeurs uniques
                return "regression"
        
        # Sinon → classification
        return "classification"
    
    def _prepare_data(self, df: pd.DataFrame, target: str):
        """Prépare les données pour l'entraînement"""
        # Séparer features et target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Conserver seulement les colonnes numériques pour simplifier
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X = X[numeric_cols]
        
        # Gérer les valeurs manquantes
        X = X.fillna(X.mean())
        
        feature_names = X.columns.tolist()
        
        return X.values, y.values, feature_names
    
    def _train_multiple_models(self, X_train, X_test, y_train, y_test, problem_type, feature_names):
        """Entraîne plusieurs modèles et retourne les résultats"""
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
            # Entraîner
            model.fit(X_train, y_train)
            
            # Prédire
            y_pred = model.predict(X_test)
            
            # Évaluer
            if problem_type == "regression":
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                score = r2
                metrics = {
                    'R²': round(r2, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4)
                }
            else:
                accuracy = accuracy_score(y_test, y_pred)
                
                score = accuracy
                metrics = {
                    'Accuracy': round(accuracy, 4)
                }
                
                # Ajouter precision/recall/f1 si binaire
                if len(np.unique(y_train)) == 2:
                    metrics['Precision'] = round(precision_score(y_test, y_pred, average='binary'), 4)
                    metrics['Recall'] = round(recall_score(y_test, y_pred, average='binary'), 4)
                    metrics['F1-Score'] = round(f1_score(y_test, y_pred, average='binary'), 4)
            
            # Feature importance si disponible
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(feature_names, np.abs(model.coef_).flatten()))
            
            results.append({
                'name': name,
                'score': score,
                'metrics': metrics,
                'feature_importance': importance
            })
            
            if score > best_score:
                best_score = score
                best_model_name = name
                self.best_model = model
                self.feature_importance = importance
        
        # Trier par score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Marquer le meilleur
        for r in results:
            r['is_best'] = (r['name'] == best_model_name)
        
        return results
    
    def _format_training_results(self, results, problem_type, target):
        """Formate les résultats en markdown"""
        output = f"# Résultats de l'Entraînement\n\n"
        output += f"**Variable cible:** {target}\n"
        output += f"**Type:** {'Régression (prédiction de nombre)' if problem_type == 'regression' else 'Classification (prédiction de catégorie)'}\n\n"
        
        output += "## Comparaison des Modèles\n\n"
        
        # Tableau des résultats
        if problem_type == "regression":
            output += "| Modèle | R² | RMSE | MAE | Meilleur |\n"
            output += "|--------|-----|------|-----|----------|\n"
            for r in results:
                check = "✓" if r['is_best'] else ""
                output += f"| {r['name']} | {r['metrics']['R²']:.3f} | {r['metrics']['RMSE']:.3f} | {r['metrics']['MAE']:.3f} | {check} |\n"
        else:
            output += "| Modèle | Accuracy | Precision | Recall | F1-Score | Meilleur |\n"
            output += "|--------|----------|-----------|--------|----------|----------|\n"
            for r in results:
                check = "✓" if r['is_best'] else ""
                prec = r['metrics'].get('Precision', '-')
                rec = r['metrics'].get('Recall', '-')
                f1 = r['metrics'].get('F1-Score', '-')
                output += f"| {r['name']} | {r['metrics']['Accuracy']:.3f} | {prec if prec == '-' else f'{prec:.3f}'} | {rec if rec == '-' else f'{rec:.3f}'} | {f1 if f1 == '-' else f'{f1:.3f}'} | {check} |\n"
        
        # Meilleur modèle
        best = [r for r in results if r['is_best']][0]
        output += f"\n## Meilleur Modèle: {best['name']}\n\n"
        
        if problem_type == "regression":
            r2_pct = best['metrics']['R²'] * 100
            output += f"Le modèle peut expliquer **{r2_pct:.1f}%** de la variance de {target}.\n"
            output += f"Erreur moyenne: **{best['metrics']['MAE']:.2f}**\n\n"
        else:
            acc_pct = best['metrics']['Accuracy'] * 100
            output += f"Précision de classification: **{acc_pct:.1f}%**\n\n"
        
        # Feature importance
        if best['feature_importance']:
            output += "## Variables les Plus Importantes\n\n"
            
            # Trier par importance
            sorted_features = sorted(
                best['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5
            
            max_importance = sorted_features[0][1]
            
            for feature, importance in sorted_features:
                bar_length = int((importance / max_importance) * 20)
                bar = "█" * bar_length
                pct = (importance / sum(best['feature_importance'].values())) * 100
                output += f"**{feature}** {bar} {pct:.1f}%\n"
        
        return output
    
    def _format_ml_suggestions(self, num_numeric_cols):
        """Formate les suggestions ML"""
        result = "# Suggestion de Modélisation ML\n\n"
        
        result += "## Modèles recommandés\n\n"
        
        result += "### Pour la prédiction numérique (Régression)\n\n"
        result += "| Modèle | Type | Quand l'utiliser |\n"
        result += "|--------|------|------------------|\n"
        result += "| Régression Linéaire | Baseline | Relation linéaire simple |\n"
        result += "| Random Forest Regressor | Robuste | Données complexes, non-linéaires |\n"
        result += "| Gradient Boosting | Performance | Haute précision requise |\n\n"
        
        result += "### Pour la classification\n\n"
        result += "| Modèle | Type | Quand l'utiliser |\n"
        result += "|--------|------|------------------|\n"
        result += "| Logistic Regression | Simple | Baseline, interprétable |\n"
        result += "| Random Forest Classifier | Robuste | Données complexes |\n"
        result += "| Gradient Boosting | Performance | Production, haute précision |\n\n"
        
        return result
    
    def _handle_hello(self, message: Message):
        """Message de test"""
        status = "disponible" if SKLEARN_AVAILABLE else "indisponible (scikit-learn manquant)"
        response = f"Agent Modélisateur ML opérationnel\nScikit-learn: {status}"
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.TASK_RESPONSE,
            content={'message': response}
        ))
    
    def _send_error(self, recipient: str, error_message: str):
        """Envoie un message d'erreur"""
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver=recipient,
            message_type=MessageType.ERROR,
            content={'error': error_message}
        ))
