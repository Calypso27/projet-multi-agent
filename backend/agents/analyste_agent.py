"""Agent Analyste - Analyse exploratoire des données"""
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, Any, Optional, List, Tuple
from ..models.message import Message, MessageType
from .base_agent import BaseAgent

# Imports optionnels pour la visualisation
try:
    import matplotlib
    matplotlib.use('Agg')  # Mode non-interactif pour serveur
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


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
        self.eda_report = None

    def _eda_complet(self, df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
        """
        Effectue une Analyse Exploratoire de Données (EDA) complète et minutieuse

        Retourne:
        - str: Rapport textuel détaillé
        - Dict: Dictionnaire de visualisations en base64
        """
        if df is None or df.empty:
            return "Aucune donnée à analyser", {}

        visualizations = {}
        report = "# RAPPORT D'ANALYSE EXPLORATOIRE DES DONNÉES (EDA)\n\n"
        report += "---\n\n"

        # 1. APERÇU GÉNÉRAL
        report += "## 1. APERÇU GÉNÉRAL DES DONNÉES\n\n"
        report += self._section_apercu_general(df)

        # 2. QUALITÉ DES DONNÉES
        report += "\n## 2. QUALITÉ DES DONNÉES\n\n"
        report += self._section_qualite_donnees(df)

        # 3. ANALYSE UNIVARIÉE - VARIABLES NUMÉRIQUES
        report += "\n## 3. ANALYSE UNIVARIÉE - VARIABLES NUMÉRIQUES\n\n"
        numeric_analysis, numeric_viz = self._analyse_variables_numeriques(df)
        report += numeric_analysis
        if numeric_viz:
            visualizations.update(numeric_viz)

        # 4. ANALYSE UNIVARIÉE - VARIABLES CATÉGORIELLES
        report += "\n## 4. ANALYSE UNIVARIÉE - VARIABLES CATÉGORIELLES\n\n"
        cat_analysis, cat_viz = self._analyse_variables_categorielles(df)
        report += cat_analysis
        if cat_viz:
            visualizations.update(cat_viz)

        # 5. DÉTECTION DES OUTLIERS
        report += "\n## 5. DÉTECTION DES VALEURS ABERRANTES (OUTLIERS)\n\n"
        outlier_analysis, outlier_viz = self._detecter_outliers(df)
        report += outlier_analysis
        if outlier_viz:
            visualizations.update(outlier_viz)

        # 6. ANALYSE BIVARIÉE - CORRÉLATIONS
        report += "\n## 6. ANALYSE BIVARIÉE - CORRÉLATIONS\n\n"
        corr_analysis, corr_viz = self._analyse_correlations(df)
        report += corr_analysis
        if corr_viz:
            visualizations.update(corr_viz)

        # 7. RECOMMANDATIONS
        report += "\n## 7. RECOMMANDATIONS POUR LA SUITE\n\n"
        report += self._generer_recommandations(df)

        # 8. CONCLUSION
        report += "\n## 8. CONCLUSION DE L'EDA\n\n"
        report += self._generer_conclusion(df)

        self.eda_report = report
        return report, visualizations

    def handle_message(self, message: Message):
        """Traite les messages"""

        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            dataset = message.content.get("dataset")

            if task == "eda_complet":
                result, visualizations = self._eda_complet(dataset)
                self.send_message(
                    receiver=message.sender,
                    message_type=MessageType.TASK_RESPONSE,
                    content={"task": task, "result": result, "visualizations": visualizations},
                    conversation_id=message.conversation_id
                )

            elif task == "analyse_complete":
                result, heatmap_data = self._analyse_complete_with_heatmap(dataset)
                self.send_message(
                    receiver=message.sender,
                    message_type=MessageType.TASK_RESPONSE,
                    content={"task": task, "result": result, "heatmap": heatmap_data},
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
    
    def _analyse_complete_with_heatmap(self, df: pd.DataFrame):
        """Analyse complète du dataset avec heatmap de corrélations"""
        if df is None or df.empty:
            return "Aucune donnée à analyser", None

        # Générer le texte d'analyse
        text_result = self._analyse_complete(df)

        # Générer le heatmap de corrélations
        heatmap_data = self._generate_correlation_heatmap(df)

        return text_result, heatmap_data

    def _generate_correlation_heatmap(self, df: pd.DataFrame):
        """Génère un heatmap des corrélations et le retourne sous forme de données"""
        if not VISUALIZATION_AVAILABLE:
            return None

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) < 2:
            return None

        try:
            # Calculer la matrice de corrélation
            corr_matrix = df[numeric_cols].corr()

            # Créer la figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # Créer le heatmap avec seaborn
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax,
                vmin=-1,
                vmax=1
            )

            ax.set_title('Matrice de Corrélation', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()

            # Convertir en base64 pour le transmettre
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            print(f"Erreur lors de la génération du heatmap: {e}")
            return None

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

    # ========== MÉTHODES POUR L'EDA COMPLET ==========

    def _section_apercu_general(self, df: pd.DataFrame) -> str:
        """Section 1: Aperçu général"""
        result = f"**Dimensions:** {df.shape[0]:,} lignes × {df.shape[1]} colonnes\n\n"

        result += "### 1.1 Types de variables\n\n"
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        bool_cols = df.select_dtypes(include=['bool']).columns

        result += "| Type | Nombre | Pourcentage |\n"
        result += "|------|--------|-------------|\n"
        result += f"| Numériques | {len(numeric_cols)} | {len(numeric_cols)/df.shape[1]*100:.1f}% |\n"
        result += f"| Catégorielles | {len(categorical_cols)} | {len(categorical_cols)/df.shape[1]*100:.1f}% |\n"
        result += f"| Dates/Temps | {len(datetime_cols)} | {len(datetime_cols)/df.shape[1]*100:.1f}% |\n"
        result += f"| Booléens | {len(bool_cols)} | {len(bool_cols)/df.shape[1]*100:.1f}% |\n\n"

        result += "### 1.2 Aperçu des premières lignes\n\n"
        preview = df.head(5)
        result += "| " + " | ".join(str(c) for c in preview.columns) + " |\n"
        result += "|" + "|".join(["---"] * len(preview.columns)) + "|\n"

        for idx, row in preview.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("N/A")
                elif isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    val_str = str(val)
                    values.append(val_str[:50] if len(val_str) > 50 else val_str)
            result += "| " + " | ".join(values) + " |\n"

        return result + "\n"

    def _section_qualite_donnees(self, df: pd.DataFrame) -> str:
        """Section 2: Qualité des données"""
        result = "### 2.1 Valeurs manquantes\n\n"

        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100

        if missing.sum() == 0:
            result += "✓ **Aucune valeur manquante détectée** - Excellente qualité!\n\n"
        else:
            result += "| Colonne | Valeurs manquantes | Pourcentage |\n"
            result += "|---------|-------------------|-------------|\n"
            for col in missing[missing > 0].index:
                result += f"| {col} | {missing[col]:,} | {missing_pct[col]:.2f}% |\n"
            result += f"\n**Total:** {missing.sum():,} valeurs manquantes ({(missing.sum()/(df.shape[0]*df.shape[1])*100):.2f}% des données)\n\n"

        result += "### 2.2 Doublons\n\n"
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            result += "✓ **Aucun doublon détecté**\n\n"
        else:
            dup_pct = (duplicates / len(df)) * 100
            result += f"⚠️ **{duplicates:,} lignes dupliquées** ({dup_pct:.2f}% des données)\n\n"

        return result + "\n"

    def _analyse_variables_numeriques(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Section 3: Analyse des variables numériques"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) == 0:
            return "Aucune variable numérique dans le dataset.\n", {}

        result = f"**{len(numeric_cols)} variables numériques** détectées.\n\n"

        result += "### 3.1 Statistiques descriptives détaillées\n\n"
        result += "| Variable | Moyenne | Médiane | Écart-type | Min | Max |\n"
        result += "|----------|---------|---------|------------|-----|-----|\n"

        for col in numeric_cols:
            mean = df[col].mean()
            median = df[col].median()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()

            result += f"| {col} | {mean:.2f} | {median:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} |\n"

        result += "\n### 3.2 Interprétation de la distribution\n\n"
        for col in numeric_cols[:5]:  # Limiter à 5 variables
            skew = df[col].skew()
            if abs(skew) < 0.5:
                distrib = "**symétrique**"
            elif skew > 0.5:
                distrib = "**asymétrique à droite**"
            else:
                distrib = "**asymétrique à gauche**"

            result += f"- **{col}**: Distribution {distrib}\n"

        # Générer les visualisations
        visualizations = {}
        if VISUALIZATION_AVAILABLE:
            # Histogrammes
            hist_img = self._generer_histogrammes(df, numeric_cols)
            if hist_img:
                visualizations['histogrammes'] = hist_img

        return result + "\n", visualizations

    def _analyse_variables_categorielles(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Section 4: Analyse des variables catégorielles"""
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) == 0:
            return "Aucune variable catégorielle dans le dataset.\n", {}

        result = f"**{len(categorical_cols)} variables catégorielles** détectées.\n\n"

        result += "### 4.1 Analyse de cardinalité\n\n"
        result += "| Variable | Valeurs uniques | Mode | Fréquence mode |\n"
        result += "|----------|-----------------|------|----------------|\n"

        for col in categorical_cols:
            unique_count = df[col].nunique()
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            mode_freq = (df[col] == mode_val).sum() if mode_val != "N/A" else 0
            mode_pct = (mode_freq / len(df) * 100) if mode_val != "N/A" else 0

            mode_str = str(mode_val)[:30]
            result += f"| {col} | {unique_count} | {mode_str} | {mode_pct:.1f}% |\n"

        result += "\n### 4.2 Distribution des catégories (Top 5 par variable)\n\n"
        for col in categorical_cols[:5]:  # Limiter à 5 variables
            result += f"**{col}:**\n\n"
            top_5 = df[col].value_counts().head(5)
            result += "| Valeur | Fréquence | Pourcentage |\n"
            result += "|--------|-----------|-------------|\n"
            for val, count in top_5.items():
                pct = (count / len(df)) * 100
                val_str = str(val)[:40]
                result += f"| {val_str} | {count:,} | {pct:.2f}% |\n"
            result += "\n"

        # Générer les visualisations
        visualizations = {}
        if VISUALIZATION_AVAILABLE:
            bar_img = self._generer_barplots(df, categorical_cols)
            if bar_img:
                visualizations['barplots'] = bar_img

        return result, visualizations

    def _detecter_outliers(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Section 5: Détection des outliers"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) == 0:
            return "Aucune variable numérique pour détecter les outliers.\n", {}

        result = "### 5.1 Méthode IQR (Interquartile Range)\n\n"
        result += "Les outliers sont détectés comme les valeurs < Q1 - 1.5×IQR ou > Q3 + 1.5×IQR\n\n"
        result += "| Variable | Outliers inférieurs | Outliers supérieurs | Total outliers | % |\n"
        result += "|----------|---------------------|---------------------|----------------|----|\n"

        outlier_summary = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            lower_outliers = (df[col] < lower_bound).sum()
            upper_outliers = (df[col] > upper_bound).sum()
            total_outliers = lower_outliers + upper_outliers
            pct = (total_outliers / len(df)) * 100

            outlier_summary[col] = {
                'lower': lower_outliers,
                'upper': upper_outliers,
                'total': total_outliers,
                'pct': pct
            }

            result += f"| {col} | {lower_outliers} | {upper_outliers} | {total_outliers} | {pct:.2f}% |\n"

        # Générer les visualisations
        visualizations = {}
        if VISUALIZATION_AVAILABLE:
            box_img = self._generer_boxplots(df, numeric_cols)
            if box_img:
                visualizations['boxplots'] = box_img

        return result + "\n", visualizations

    def _analyse_correlations(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Section 6: Analyse des corrélations"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) < 2:
            return "Pas assez de variables numériques pour calculer les corrélations.\n", {}

        corr_matrix = df[numeric_cols].corr()

        result = "### 6.1 Corrélations fortes (|r| > 0.7)\n\n"

        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

        if strong_corr:
            result += "| Variable 1 | Variable 2 | Corrélation | Type |\n"
            result += "|------------|------------|-------------|------|\n"
            for col1, col2, val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                corr_type = "Positive forte" if val > 0 else "Négative forte"
                result += f"| {col1} | {col2} | {val:.3f} | {corr_type} |\n"
        else:
            result += "✓ Aucune corrélation forte détectée - Les variables sont relativement indépendantes.\n"

        result += "\n### 6.2 Corrélations modérées (0.5 < |r| < 0.7)\n\n"
        moderate_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if 0.5 < abs(val) <= 0.7:
                    moderate_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

        if moderate_corr:
            result += "| Variable 1 | Variable 2 | Corrélation |\n"
            result += "|------------|------------|-------------|\n"
            for col1, col2, val in sorted(moderate_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
                result += f"| {col1} | {col2} | {val:.3f} |\n"
        else:
            result += "Aucune corrélation modérée détectée.\n"

        # Générer le heatmap
        visualizations = {}
        if VISUALIZATION_AVAILABLE:
            heatmap_img = self._generate_correlation_heatmap(df)
            if heatmap_img:
                visualizations['correlation_heatmap'] = heatmap_img

        return result + "\n", visualizations

    def _generer_recommandations(self, df: pd.DataFrame) -> str:
        """Section 7: Recommandations"""
        result = ""
        recommendations = []

        # Valeurs manquantes
        missing = df.isnull().sum().sum()
        if missing > 0:
            missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
            if missing_pct > 5:
                recommendations.append(f"🔴 **CRITIQUE**: {missing_pct:.1f}% de données manquantes - Imputation ou suppression nécessaire")
            else:
                recommendations.append(f"🟡 **ATTENTION**: {missing_pct:.1f}% de données manquantes - À traiter avant modélisation")

        # Doublons
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            recommendations.append(f"🟡 **Doublons**: {dup_pct:.1f}% de lignes dupliquées - Vérifier si c'est intentionnel")

        # Outliers
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        total_outliers = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            total_outliers += outliers

        if total_outliers > 0:
            outlier_pct = (total_outliers / (len(df) * len(numeric_cols))) * 100
            recommendations.append(f"🟡 **Outliers**: {total_outliers:,} valeurs aberrantes détectées - Analyser et traiter si nécessaire")

        # Variables très corrélées
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            high_corr_count = 0
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_count += 1

            if high_corr_count > 0:
                recommendations.append(f"🟡 **Multicollinéarité**: {high_corr_count} paires de variables très corrélées (>0.9) - Envisager de supprimer des variables redondantes")

        # Cardinalité élevée
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                recommendations.append(f"🟡 **Haute cardinalité**: Variable '{col}' a {df[col].nunique()} valeurs uniques - Envisager l'encodage ou le regroupement")

        # Distribution asymétrique
        skewed_vars = []
        for col in numeric_cols:
            if abs(df[col].skew()) > 2:
                skewed_vars.append(col)

        if skewed_vars:
            recommendations.append(f"🟢 **Transformation**: {len(skewed_vars)} variables très asymétriques - Envisager log/sqrt transformation")

        if not recommendations:
            result += "✅ **Données de bonne qualité** - Aucune action critique requise!\n\n"
        else:
            result += "### Actions recommandées:\n\n"
            for i, rec in enumerate(recommendations, 1):
                result += f"{i}. {rec}\n"

        return result + "\n"

    def _generer_conclusion(self, df: pd.DataFrame) -> str:
        """Section 8: Conclusion"""
        result = ""

        # Résumé global
        result += "### Résumé exécutif\n\n"
        result += f"- **Dataset**: {df.shape[0]:,} observations × {df.shape[1]} variables\n"

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        result += f"- **Composition**: {len(numeric_cols)} numériques, {len(categorical_cols)} catégorielles\n"

        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        quality = "excellente" if missing_pct < 1 else "bonne" if missing_pct < 5 else "moyenne" if missing_pct < 15 else "faible"
        result += f"- **Qualité globale**: {quality.capitalize()} ({missing_pct:.2f}% de données manquantes)\n"

        result += "\n### Prochaines étapes suggérées\n\n"
        result += "1. **Nettoyage**: Traiter les valeurs manquantes et outliers identifiés\n"
        result += "2. **Feature Engineering**: Créer de nouvelles variables si pertinent\n"
        result += "3. **Encodage**: Transformer les variables catégorielles pour la modélisation\n"
        result += "4. **Normalisation**: Standardiser les variables numériques si nécessaire\n"
        result += "5. **Modélisation**: Passer à la phase de Machine Learning\n"

        result += "\n---\n\n"
        result += "*Rapport généré automatiquement par l'Agent Analyste EDA*\n"

        return result

    # ========== MÉTHODES DE VISUALISATION ==========

    def _generer_histogrammes(self, df: pd.DataFrame, numeric_cols) -> Optional[str]:
        """Génère des histogrammes pour les variables numériques"""
        if not VISUALIZATION_AVAILABLE or len(numeric_cols) == 0:
            return None

        try:
            n_cols = min(len(numeric_cols), 6)  # Maximum 6 variables
            cols_to_plot = numeric_cols[:n_cols]

            n_rows = (n_cols + 2) // 3
            n_plot_cols = min(n_cols, 3)

            fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(15, 5*n_rows))
            if n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten() if n_cols > 1 else axes

            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution de {col}', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Fréquence')
                ax.grid(True, alpha=0.3)

            # Masquer les axes inutilisés
            for idx in range(len(cols_to_plot), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            print(f"Erreur génération histogrammes: {e}")
            return None

    def _generer_boxplots(self, df: pd.DataFrame, numeric_cols) -> Optional[str]:
        """Génère des boxplots pour détecter les outliers"""
        if not VISUALIZATION_AVAILABLE or len(numeric_cols) == 0:
            return None

        try:
            n_cols = min(len(numeric_cols), 6)
            cols_to_plot = numeric_cols[:n_cols]

            n_rows = (n_cols + 2) // 3
            n_plot_cols = min(n_cols, 3)

            fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(15, 5*n_rows))
            if n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten() if n_cols > 1 else axes

            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                df.boxplot(column=col, ax=ax)
                ax.set_title(f'Boxplot de {col}', fontweight='bold')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)

            for idx in range(len(cols_to_plot), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            print(f"Erreur génération boxplots: {e}")
            return None

    def _generer_barplots(self, df: pd.DataFrame, categorical_cols) -> Optional[str]:
        """Génère des barplots pour les variables catégorielles"""
        if not VISUALIZATION_AVAILABLE or len(categorical_cols) == 0:
            return None

        try:
            n_cols = min(len(categorical_cols), 4)
            cols_to_plot = categorical_cols[:n_cols]

            n_rows = (n_cols + 1) // 2
            n_plot_cols = min(n_cols, 2)

            fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(14, 5*n_rows))
            if n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten() if n_cols > 1 else axes

            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                top_10 = df[col].value_counts().head(10)
                top_10.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(f'Top 10 valeurs de {col}', fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Fréquence')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')

            for idx in range(len(cols_to_plot), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            print(f"Erreur génération barplots: {e}")
            return None
