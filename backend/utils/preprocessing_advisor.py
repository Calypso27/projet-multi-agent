"""
Conseiller de preprocessing intelligent.
Analyse le dataset et génère un plan de preprocessing basé sur les caractéristiques réelles des données.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def _missing_strategy_numeric(col: pd.Series) -> Tuple[str, str]:
    """Choisit la stratégie d'imputation pour une colonne numérique."""
    missing_pct = col.isnull().mean() * 100
    if missing_pct == 0:
        return 'none', 'Aucune valeur manquante'
    if missing_pct > 60:
        return 'drop_column', f'{missing_pct:.1f}% de valeurs manquantes — colonne non exploitable'
    non_null = col.dropna()
    if len(non_null) < 5:
        return 'median', 'Trop peu de valeurs — médiane sécurisée'
    skewness = abs(float(non_null.skew()))
    if skewness > 1.0:
        return 'median', f'Distribution asymétrique (skewness={skewness:.2f}) — médiane robuste'
    else:
        return 'mean', f'Distribution symétrique (skewness={skewness:.2f}) — moyenne appropriée'


def _outlier_analysis(col: pd.Series) -> Tuple[str, str, float]:
    """
    Analyse les outliers combinant IQR (robustesse) et Z-score (magnitude).

    Logique de décision :
      - Aucun outlier                     → keep
      - Déviation extrême (> 3×IQR)       → cap  (erreur de saisie probable)
      - < 15% outliers                    → cap  (bruit, à corriger)
      - >= 15% outliers, déviation faible → keep (distribution réelle, ne pas altérer)
    """
    non_null = col.dropna()
    if len(non_null) < 10:
        return 'keep', 'Données insuffisantes pour analyse outliers', 0.0

    Q1    = float(non_null.quantile(0.25))
    Q3    = float(non_null.quantile(0.75))
    IQR   = Q3 - Q1
    if IQR == 0:
        return 'keep', 'Pas de dispersion (IQR=0)', 0.0

    lower_mild    = Q1 - 1.5 * IQR   # seuil standard
    upper_mild    = Q3 + 1.5 * IQR
    lower_extreme = Q1 - 3.0 * IQR   # seuil extrême
    upper_extreme = Q3 + 3.0 * IQR

    mask_mild    = (non_null < lower_mild)    | (non_null > upper_mild)
    mask_extreme = (non_null < lower_extreme) | (non_null > upper_extreme)

    outlier_pct         = float(mask_mild.mean() * 100)
    extreme_outlier_pct = float(mask_extreme.mean() * 100)

    if outlier_pct == 0:
        return 'keep', 'Aucun outlier détecté (IQR 1.5×)', 0.0

    # Outliers extrêmes (> 3×IQR) = très probablement des erreurs de saisie
    if extreme_outlier_pct > 0:
        max_val = float(non_null.max())
        min_val = float(non_null.min())
        return (
            'cap',
            f'{outlier_pct:.1f}% outliers dont {extreme_outlier_pct:.1f}% extrêmes (> 3×IQR) '
            f'— erreurs probables [min={min_val:.4g}, max={max_val:.4g}] — écrêtage IQR 1.5×',
            outlier_pct
        )

    # Outliers modérés (1.5×IQR) : cap si < 15%, sinon distribution réelle
    if outlier_pct < 15:
        return (
            'cap',
            f'{outlier_pct:.1f}% outliers modérés (1.5×IQR) — bruit probable — écrêtage',
            outlier_pct
        )
    else:
        return (
            'keep',
            f'{outlier_pct:.1f}% outliers (1.5×IQR) — proportion élevée, distribution potentiellement naturelle — conservés',
            outlier_pct
        )


def _detect_correlated_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """Détecte les paires de colonnes hautement corrélées."""
    numeric = df.select_dtypes(include=['number'])
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr.iloc[i, j])
            if val >= threshold:
                pairs.append((cols[j], cols[i], val))  # (à_supprimer, à_garder, corrélation)
    return pairs


def _detect_low_variance(df: pd.DataFrame) -> List[str]:
    """Détecte les colonnes avec variance quasi-nulle."""
    low_var = []
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].nunique() <= 1:
            low_var.append(col)
        elif df[col].nunique() < 5:
            top_freq = df[col].value_counts(normalize=True).iloc[0]
            if top_freq > 0.98:
                low_var.append(col)
    return low_var


def build_preprocessing_plan(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Analyse experte du dataset et génère un plan de preprocessing structuré.

    Retourne:
        {
            'columns': {col: {dtype, missing_pct, action_missing, reason_missing,
                               action_outliers, reason_outliers, outlier_pct, drop, drop_reason}},
            'global': {correlated_pairs, low_variance_cols, recommended_drop},
            'summary': {total_columns, cols_with_missing, cols_with_outliers, cols_to_drop, ...},
            'target_col': str
        }
    """
    plan: Dict[str, Any] = {
        'columns': {},
        'global': {
            'correlated_pairs': [],
            'low_variance_cols': [],
            'recommended_drop': [],
        },
        'summary': {},
        'target_col': target_col,
    }

    # ── Analyse colonne par colonne ────────────────────────────────────────────
    for col in df.columns:
        is_numeric = df[col].dtype.kind in ('i', 'f', 'u')
        missing_pct = round(float(df[col].isnull().mean() * 100), 2)
        n_unique = int(df[col].nunique())

        info: Dict[str, Any] = {
            'dtype': str(df[col].dtype),
            'missing_pct': missing_pct,
            'n_unique': n_unique,
            'action_missing': 'none',
            'reason_missing': 'Aucune valeur manquante',
            'action_outliers': 'keep',
            'reason_outliers': 'Non numérique ou sans outlier',
            'outlier_pct': 0.0,
            'drop': False,
            'drop_reason': '',
        }

        # Stratégie valeurs manquantes
        if missing_pct > 0:
            if is_numeric:
                action, reason = _missing_strategy_numeric(df[col])
            else:
                if missing_pct > 60:
                    action = 'drop_column'
                    reason = f'{missing_pct:.1f}% manquants — non exploitable'
                else:
                    action = 'mode'
                    reason = f'Variable catégorielle — imputation par la valeur la plus fréquente ({missing_pct:.1f}% manquants)'
            info['action_missing'] = action
            info['reason_missing'] = reason
            if action == 'drop_column':
                info['drop'] = True
                info['drop_reason'] = reason

        # Analyse outliers (numériques uniquement, hors target)
        if is_numeric and col != target_col and not info['drop']:
            action_o, reason_o, pct_o = _outlier_analysis(df[col])
            info['action_outliers'] = action_o
            info['reason_outliers'] = reason_o
            info['outlier_pct'] = pct_o

        plan['columns'][col] = info

    # ── Analyse globale : multicolinéarité ─────────────────────────────────────
    corr_pairs = _detect_correlated_columns(df)
    plan['global']['correlated_pairs'] = corr_pairs
    for col_drop, col_keep, corr_val in corr_pairs:
        if col_drop != target_col and col_drop not in plan['global']['recommended_drop']:
            plan['global']['recommended_drop'].append(col_drop)
            if col_drop in plan['columns']:
                plan['columns'][col_drop]['drop'] = True
                plan['columns'][col_drop]['drop_reason'] = (
                    f'Multicolinéarité : corrélée à {col_keep} (r={corr_val:.2f}) — redondante'
                )

    # ── Analyse globale : variance nulle ──────────────────────────────────────
    low_var = _detect_low_variance(df)
    plan['global']['low_variance_cols'] = low_var
    for col in low_var:
        if col != target_col and col not in plan['global']['recommended_drop']:
            plan['global']['recommended_drop'].append(col)
            if col in plan['columns']:
                plan['columns'][col]['drop'] = True
                plan['columns'][col]['drop_reason'] = 'Variance quasi-nulle — apport prédictif nul'

    # ── Résumé ────────────────────────────────────────────────────────────────
    plan['summary'] = {
        'total_columns': len(df.columns),
        'cols_with_missing': sum(1 for c in plan['columns'].values() if c['missing_pct'] > 0),
        'cols_with_outliers': sum(1 for c in plan['columns'].values() if c['outlier_pct'] > 0),
        'cols_to_drop': sum(1 for c in plan['columns'].values() if c['drop']),
        'correlated_pairs': len(corr_pairs),
        'low_variance': len(low_var),
    }

    return plan
