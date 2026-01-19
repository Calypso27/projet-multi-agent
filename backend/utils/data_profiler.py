"""Profiler de données automatique"""
import pandas as pd
from typing import Dict, List, Any


class DataProfiler:
    """Analyse automatique d'un dataset"""
    
    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Génère un profil complet du dataset"""
        
        profile = {
            'dimensions': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'column_types': DataProfiler._analyze_column_types(df),
            'missing_values': DataProfiler._analyze_missing_values(df),
            'duplicates': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'date_columns': DataProfiler._detect_date_columns(df),
            'suggestions': DataProfiler._generate_suggestions(df)
        }
        
        return profile
    
    @staticmethod
    def _analyze_column_types(df: pd.DataFrame) -> Dict[str, int]:
        """Compte les types de colonnes"""
        numeric = len(df.select_dtypes(include=['int64', 'float64']).columns)
        categorical = len(df.select_dtypes(include=['object']).columns)
        datetime = len(df.select_dtypes(include=['datetime64']).columns)
        
        return {
            'numeric': numeric,
            'categorical': categorical,
            'datetime': datetime
        }
    
    @staticmethod
    def _analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse des valeurs manquantes"""
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        
        missing_by_column = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_by_column[col] = {
                    'count': int(missing_count),
                    'percentage': round((missing_count / len(df)) * 100, 2)
                }
        
        return {
            'total': int(total_missing),
            'percentage': round((total_missing / total_cells) * 100, 2),
            'by_column': missing_by_column
        }
    
    @staticmethod
    def _detect_date_columns(df: pd.DataFrame) -> List[str]:
        """Détecte les colonnes de dates"""
        date_cols = []
        
        date_cols.extend(df.select_dtypes(include=['datetime64']).columns.tolist())
        
        for col in df.select_dtypes(include=['object']).columns:
            if DataProfiler._is_likely_date(df[col]):
                date_cols.append(col)
        
        return date_cols
    
    @staticmethod
    def _is_likely_date(series: pd.Series) -> bool:
        """Vérifie si une série ressemble à des dates"""
        if len(series) == 0:
            return False
        
        sample = series.dropna().head(10)
        
        try:
            pd.to_datetime(sample)
            return True
        except:
            return False
    
    @staticmethod
    def _generate_suggestions(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Génère des suggestions automatiques"""
        suggestions = []
        
        date_cols = DataProfiler._detect_date_columns(df)
        if date_cols:
            suggestions.append({
                'type': 'timeseries',
                'title': 'Analyse temporelle disponible',
                'description': f'Colonne(s) date détectée(s): {", ".join(date_cols)}',
                'action': 'plot_timeseries'
            })
        
        missing = df.isnull().sum().sum()
        if missing > 0:
            pct = (missing / (df.shape[0] * df.shape[1])) * 100
            suggestions.append({
                'type': 'missing',
                'title': 'Valeurs manquantes détectées',
                'description': f'{missing} valeurs manquantes ({pct:.1f}%)',
                'action': 'handle_missing'
            })
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'prediction',
                'title': 'Prédiction possible',
                'description': f'{len(numeric_cols)} colonnes numériques disponibles',
                'action': 'start_prediction'
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'correlation',
                'title': 'Analyse de corrélations',
                'description': 'Découvrir les relations entre variables',
                'action': 'show_correlations'
            })
        
        return suggestions
    
    @staticmethod
    def get_data_type_description(df: pd.DataFrame) -> str:
        """Retourne une description textuelle du type de données"""
        profile = DataProfiler.profile(df)
        
        if profile['date_columns']:
            return "Données de séries temporelles"
        elif profile['column_types']['numeric'] > profile['column_types']['categorical']:
            return "Données numériques"
        elif profile['column_types']['categorical'] > 0:
            return "Données mixtes (numériques et catégorielles)"
        else:
            return "Données tabulaires"
