"""Module de preprocessing des données - Version intelligente"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Classe utilitaire pour le preprocessing automatique des données

    Fonctionnalités:
    - Nettoyage (NaN, doublons, outliers)
    - Encodage (One-Hot, Label)
    - Normalisation (StandardScaler)
    - Train/Test Split
    """

    @staticmethod
    def clean_data(df: pd.DataFrame,
                   remove_duplicates: bool = True,
                   handle_missing: str = 'auto',
                   remove_outliers: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Nettoie le dataset

        Args:
            df: DataFrame à nettoyer
            remove_duplicates: Supprimer les doublons
            handle_missing: 'auto', 'drop', 'median', 'mean', 'mode'
            remove_outliers: Supprimer les outliers (méthode IQR)

        Returns:
            (df_clean, rapport_nettoyage)
        """
        df_clean = df.copy()
        rapport = {
            'rows_before': len(df),
            'rows_after': 0,
            'duplicates_removed': 0,
            'missing_handled': {},
            'outliers_removed': 0
        }

        # 1. Suppression des doublons
        if remove_duplicates:
            duplicates_count = df_clean.duplicated().sum()
            if duplicates_count > 0:
                df_clean = df_clean.drop_duplicates()
                rapport['duplicates_removed'] = duplicates_count

        # 2. Gestion des valeurs manquantes
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()

        if missing_cols:
            for col in missing_cols:
                missing_count = df_clean[col].isnull().sum()
                missing_pct = (missing_count / len(df_clean)) * 100

                # Stratégie automatique
                if handle_missing == 'auto':
                    # Si plus de 50% de NaN, supprimer la colonne
                    if missing_pct > 50:
                        df_clean = df_clean.drop(columns=[col])
                        rapport['missing_handled'][col] = f"Colonne supprimée ({missing_pct:.1f}% NaN)"
                    # Sinon, imputation selon le type
                    elif df_clean[col].dtype in ['int64', 'float64']:
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        rapport['missing_handled'][col] = f"Imputation médiane ({missing_count} valeurs)"
                    else:
                        mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'UNKNOWN'
                        df_clean[col] = df_clean[col].fillna(mode_val)
                        rapport['missing_handled'][col] = f"Imputation mode ({missing_count} valeurs)"

                elif handle_missing == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                    rapport['missing_handled'][col] = f"Lignes supprimées ({missing_count} valeurs)"

                elif handle_missing in ['median', 'mean'] and df_clean[col].dtype in ['int64', 'float64']:
                    fill_val = df_clean[col].median() if handle_missing == 'median' else df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    rapport['missing_handled'][col] = f"Imputation {handle_missing} ({missing_count} valeurs)"

                elif handle_missing == 'mode':
                    mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'UNKNOWN'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    rapport['missing_handled'][col] = f"Imputation mode ({missing_count} valeurs)"

        # 3. Suppression des outliers (IQR) - uniquement colonnes numériques
        if remove_outliers:
            numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
            rows_before = len(df_clean)

            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filtrer
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

            rapport['outliers_removed'] = rows_before - len(df_clean)

        rapport['rows_after'] = len(df_clean)

        return df_clean, rapport

    @staticmethod
    def encode_categorical(df: pd.DataFrame,
                          max_categories: int = 10,
                          drop_first: bool = False) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Encode les variables catégorielles

        Stratégie:
        - One-Hot Encoding si <= max_categories valeurs uniques
        - Label Encoding sinon

        Args:
            df: DataFrame
            max_categories: Seuil pour One-Hot vs Label Encoding
            drop_first: Éviter la multicolinéarité pour One-Hot

        Returns:
            (df_encoded, rapport_encodage)
        """
        df_encoded = df.copy()
        rapport = {}

        categorical_cols = df_encoded.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()

            # One-Hot Encoding pour peu de catégories
            if unique_count <= max_categories:
                # Créer les colonnes One-Hot
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])

                rapport[col] = f"One-Hot Encoding ({unique_count} catégories → {len(dummies.columns)} colonnes)"

            # Label Encoding pour beaucoup de catégories
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                rapport[col] = f"Label Encoding ({unique_count} catégories → 0-{unique_count-1})"

        return df_encoded, rapport

    @staticmethod
    def normalize_features(df: pd.DataFrame,
                          exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalise les variables numériques avec StandardScaler

        Args:
            df: DataFrame
            exclude_cols: Colonnes à exclure de la normalisation

        Returns:
            (df_normalized, rapport_normalisation)
        """
        df_normalized = df.copy()
        rapport = {
            'scaler_used': 'StandardScaler',
            'columns_normalized': []
        }

        exclude_cols = exclude_cols or []
        numeric_cols = df_normalized.select_dtypes(include=['int64', 'float64']).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        if len(cols_to_normalize) > 0:
            scaler = StandardScaler()
            df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])
            rapport['columns_normalized'] = cols_to_normalize.tolist()

        return df_normalized, rapport

    @staticmethod
    def train_test_split_data(df: pd.DataFrame,
                             target_col: Optional[str] = None,
                             test_size: float = 0.2,
                             random_state: int = 42,
                             stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split train/test avec stratification optionnelle

        Args:
            df: DataFrame
            target_col: Colonne cible (pour stratification)
            test_size: Proportion du test set
            random_state: Seed pour reproductibilité
            stratify: Utiliser stratification si target_col fournie

        Returns:
            (df_train, df_test, rapport_split)
        """
        rapport = {
            'train_size': 0,
            'test_size': 0,
            'test_ratio': test_size,
            'stratified': False,
            'target_column': target_col
        }

        # Vérifier si stratification possible
        stratify_col = None
        if target_col and target_col in df.columns and stratify:
            # Stratification uniquement pour classification (peu de valeurs uniques)
            if df[target_col].nunique() < 50:
                stratify_col = df[target_col]
                rapport['stratified'] = True

        # Split
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        rapport['train_size'] = len(df_train)
        rapport['test_size'] = len(df_test)

        return df_train, df_test, rapport

    @staticmethod
    def preprocess_pipeline(df: pd.DataFrame,
                          target_col: Optional[str] = None,
                          clean: bool = True,
                          encode: bool = True,
                          normalize: bool = True,
                          split: bool = True,
                          test_size: float = 0.2) -> Dict[str, Any]:
        """
        Pipeline complet de preprocessing

        Args:
            df: DataFrame original
            target_col: Colonne cible (pour split)
            clean: Activer nettoyage
            encode: Activer encodage
            normalize: Activer normalisation
            split: Activer train/test split
            test_size: Taille du test set

        Returns:
            Dict avec:
            - 'data_train': DataFrame train (ou df complet si pas de split)
            - 'data_test': DataFrame test (ou None)
            - 'rapport': Rapport complet du preprocessing
        """
        result = {
            'data_train': None,
            'data_test': None,
            'rapport': {
                'steps_executed': [],
                'cleaning': {},
                'encoding': {},
                'normalization': {},
                'split': {}
            }
        }

        df_processed = df.copy()

        # 1. Nettoyage
        if clean:
            df_processed, rapport_clean = DataPreprocessor.clean_data(
                df_processed,
                remove_duplicates=True,
                handle_missing='auto',
                remove_outliers=False
            )
            result['rapport']['cleaning'] = rapport_clean
            result['rapport']['steps_executed'].append('Nettoyage')

        # 2. Encodage
        if encode:
            df_processed, rapport_encode = DataPreprocessor.encode_categorical(
                df_processed,
                max_categories=10
            )
            result['rapport']['encoding'] = rapport_encode
            result['rapport']['steps_executed'].append('Encodage')

        # 3. Normalisation
        if normalize:
            # Exclure la target de la normalisation
            exclude = [target_col] if target_col and target_col in df_processed.columns else []
            df_processed, rapport_norm = DataPreprocessor.normalize_features(
                df_processed,
                exclude_cols=exclude
            )
            result['rapport']['normalization'] = rapport_norm
            result['rapport']['steps_executed'].append('Normalisation')

        # 4. Train/Test Split
        if split and len(df_processed) > 50:  # Minimum 50 lignes pour split
            df_train, df_test, rapport_split = DataPreprocessor.train_test_split_data(
                df_processed,
                target_col=target_col,
                test_size=test_size
            )
            result['data_train'] = df_train
            result['data_test'] = df_test
            result['rapport']['split'] = rapport_split
            result['rapport']['steps_executed'].append('Train/Test Split')
        else:
            result['data_train'] = df_processed
            result['data_test'] = None

        return result

    @staticmethod
    def apply_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applique un plan de preprocessing expert généré par preprocessing_advisor.
        Stratégie différenciée par colonne (médiane/moyenne/mode selon distribution).
        """
        df_clean = df.copy()
        rapport: Dict[str, Any] = {
            'rows_before': len(df),
            'rows_after': 0,
            'actions': {},
            'dropped_columns': [],
            'outliers_capped': [],
        }

        cols_plan = plan.get('columns', {})

        # 1. Supprimer les colonnes marquées
        to_drop = [col for col, info in cols_plan.items()
                   if info.get('drop') and col in df_clean.columns]
        if to_drop:
            df_clean = df_clean.drop(columns=to_drop)
            rapport['dropped_columns'] = to_drop
            for col in to_drop:
                rapport['actions'][col] = f"Supprimée — {cols_plan[col].get('drop_reason', '')}"

        # 2. Imputation des valeurs manquantes
        for col, info in cols_plan.items():
            if col not in df_clean.columns:
                continue
            action = info.get('action_missing', 'none')
            if action == 'none' or df_clean[col].isnull().sum() == 0:
                continue
            missing_n = int(df_clean[col].isnull().sum())
            if action == 'median':
                val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(val)
                rapport['actions'][col] = f"Médiane={val:.4g} ({missing_n} valeurs) — {info.get('reason_missing', '')}"
            elif action == 'mean':
                val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(val)
                rapport['actions'][col] = f"Moyenne={val:.4g} ({missing_n} valeurs) — {info.get('reason_missing', '')}"
            elif action == 'mode':
                mode_vals = df_clean[col].mode()
                if len(mode_vals) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_vals[0])
                    rapport['actions'][col] = f"Mode='{mode_vals[0]}' ({missing_n} valeurs) — {info.get('reason_missing', '')}"

        # 3. Écrêtage des outliers (cap)
        for col, info in cols_plan.items():
            if col not in df_clean.columns:
                continue
            if info.get('action_outliers') == 'cap':
                Q1 = float(df_clean[col].quantile(0.25))
                Q3 = float(df_clean[col].quantile(0.75))
                IQR = Q3 - Q1
                if IQR > 0:
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    n_capped = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                    rapport['outliers_capped'].append(col)
                    existing = rapport['actions'].get(col, '')
                    rapport['actions'][col] = (existing + f" | Outliers écrêtés : {n_capped} valeurs [{lower:.4g}, {upper:.4g}]").lstrip(' | ')

        # 4. Supprimer les doublons
        dups = int(df_clean.duplicated().sum())
        if dups > 0:
            df_clean = df_clean.drop_duplicates()
            rapport['duplicates_removed'] = dups

        rapport['rows_after'] = len(df_clean)
        return df_clean, rapport
