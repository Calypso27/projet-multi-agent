"""Agent Analyste - Analyse exploratoire des données"""
import pandas as pd
from typing import Dict, Any, Optional
from ..models.message import Message, MessageType
from .base_agent import BaseAgent


class AnalysteAgent(BaseAgent):
    """
    Agent Analyste
    - Analyse exploratoire des données (EDA)
    - Statistiques descriptives
    - Détection de problèmes de qualité
    """
    
    def __init__(self):
        super().__init__(name="Analyste", role="Analyste Exploratoire")
        self.last_analysis = None
    
    def handle_message(self, message: Message):
        """Traite les messages"""
        
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            dataset = message.content.get("dataset")
            
            if task == "analyse_complete":
                result = self._analyse_complete(dataset)
                self.send_message(
                    receiver=message.sender,
                    message_type=MessageType.TASK_RESPONSE,
                    content={"task": task, "result": result},
                    conversation_id=message.conversation_id
                )
            
            elif task == "statistiques":
                result = self._statistiques_descriptives(dataset)
                self.send_message(
                    receiver=message.sender,
                    message_type=MessageType.TASK_RESPONSE,
                    content={"task": task, "result": result},
                    conversation_id=message.conversation_id
                )
            
            elif task == "resume":
                result = self._resume_dataset(dataset)
                self.send_message(
                    receiver=message.sender,
                    message_type=MessageType.TASK_RESPONSE,
                    content={"task": task, "result": result},
                    conversation_id=message.conversation_id
                )
    
    def _analyse_complete(self, df: pd.DataFrame) -> str:
        """Analyse complète du dataset"""
        if df is None or df.empty:
            return "Aucune donnée à analyser"
        
        result = "# ANALYSE COMPLETE DU DATASET\n\n"
        
        # Informations générales
        result += f"**Dimensions:** {df.shape[0]} lignes × {df.shape[1]} colonnes\n\n"
        
        # Colonnes avec tableau
        result += "## Colonnes\n\n"
        result += "| Nom | Type | Valeurs uniques |\n"
        result += "|-----|------|----------------|\n"
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            result += f"| {col} | {dtype} | {unique_count} |\n"
        
        result += "\n"
        
        # Valeurs manquantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            result += "## Valeurs manquantes\n\n"
            result += "| Colonne | Nombre | Pourcentage |\n"
            result += "|---------|--------|-------------|\n"
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                result += f"| {col} | {count} | {pct:.1f}% |\n"
        else:
            result += "## Qualité des données\n\n"
            result += "✓ Aucune valeur manquante détectée\n"
        
        result += "\n"
        
        # Statistiques pour colonnes numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            result += "## Statistiques (colonnes numériques)\n\n"
            result += "| Colonne | Moyenne | Médiane | Min | Max | Écart-type |\n"
            result += "|---------|---------|---------|-----|-----|------------|\n"
            for col in numeric_cols[:10]:  # Limiter à 10 colonnes
                result += f"| {col} | {df[col].mean():.2f} | {df[col].median():.2f} | {df[col].min():.2f} | {df[col].max():.2f} | {df[col].std():.2f} |\n"
        
        result += "\n"
        
        # Corrélations
        if len(numeric_cols) > 1:
            result += "## Corrélations principales\n\n"
            corr = df[numeric_cols].corr()
            
            # Trouver les corrélations fortes (> 0.7 ou < -0.7)
            strong_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.7:
                        strong_corr.append((corr.columns[i], corr.columns[j], val))
            
            if strong_corr:
                result += "| Variable 1 | Variable 2 | Corrélation |\n"
                result += "|------------|------------|-------------|\n"
                for col1, col2, val in strong_corr[:10]:  # Top 10
                    result += f"| {col1} | {col2} | {val:.3f} |\n"
            else:
                result += "Aucune corrélation forte (>0.7) détectée\n"
        
        self.last_analysis = result
        return result
    
    def _statistiques_descriptives(self, df: pd.DataFrame) -> str:
        """Statistiques descriptives"""
        if df is None or df.empty:
            return "Aucune donnée à analyser"
        
        result = "# STATISTIQUES DESCRIPTIVES\n\n"
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            return "Aucune colonne numérique à analyser"
        
        result += "| Colonne | Valeurs | Moyenne | Médiane | Écart-type | Min | Q1 | Q3 | Max |\n"
        result += "|---------|---------|---------|---------|------------|-----|----|----|-----|\n"
        
        for col in numeric_cols:
            count = df[col].count()
            mean = df[col].mean()
            median = df[col].median()
            std = df[col].std()
            min_val = df[col].min()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            max_val = df[col].max()
            
            result += f"| {col} | {count} | {mean:.2f} | {median:.2f} | {std:.2f} | {min_val:.2f} | {q1:.2f} | {q3:.2f} | {max_val:.2f} |\n"
        
        return result
    
    def _resume_dataset(self, df: pd.DataFrame) -> str:
        """Résumé du dataset"""
        if df is None or df.empty:
            return "Aucune donnée à analyser"
        
        result = "# RESUME DU DATASET\n\n"
        
        result += f"**Taille:** {df.shape[0]} lignes × {df.shape[1]} colonnes\n\n"
        
        # Types de colonnes
        numeric_count = len(df.select_dtypes(include=['int64', 'float64']).columns)
        text_count = len(df.select_dtypes(include=['object']).columns)
        date_count = len(df.select_dtypes(include=['datetime64']).columns)
        
        result += "## Types de colonnes\n\n"
        result += "| Type | Nombre |\n"
        result += "|------|--------|\n"
        result += f"| Numériques | {numeric_count} |\n"
        result += f"| Texte | {text_count} |\n"
        result += f"| Dates | {date_count} |\n\n"
        
        # Qualité des données
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        duplicates = df.duplicated().sum()
        
        result += "## Qualité des données\n\n"
        result += "| Métrique | Valeur |\n"
        result += "|----------|--------|\n"
        result += f"| Cellules totales | {total_cells:,} |\n"
        result += f"| Cellules manquantes | {missing_cells} ({missing_pct:.1f}%) |\n"
        result += f"| Lignes dupliquées | {duplicates} |\n\n"
        
        # Aperçu des premières lignes
        result += "## Aperçu des données (3 premières lignes)\n\n"
        
        # Créer un tableau markdown
        preview = df.head(3)
        
        # En-têtes
        result += "| " + " | ".join(preview.columns) + " |\n"
        result += "|" + "|".join(["---"] * len(preview.columns)) + "|\n"
        
        # Lignes
        for idx, row in preview.iterrows():
            values = []
            for val in row:
                # Formater les valeurs
                if pd.isna(val):
                    values.append("N/A")
                elif isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            result += "| " + " | ".join(values) + " |\n"
        
        return result
