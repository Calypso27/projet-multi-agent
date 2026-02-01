"""Data Quality Scorer - Évaluation professionnelle de la qualité des données"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class DataQualityScorer:
    """
    Évalue la qualité d'un dataset selon 5 dimensions professionnelles:
    1. Complétude (Completeness)
    2. Validité (Validity)
    3. Cohérence (Consistency)
    4. Unicité (Uniqueness)
    5. Exactitude (Accuracy - basé sur des heuristiques)
    """

    @staticmethod
    def evaluate(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Évalue la qualité globale d'un dataset

        Returns:
            Dict avec scores, détails et recommandations
        """
        results = {
            'overall_score': 0,
            'grade': '',
            'dimensions': {},
            'details': {},
            'recommendations': [],
            'summary': {}
        }

        # 1. Complétude
        completeness = DataQualityScorer._evaluate_completeness(df)
        results['dimensions']['completeness'] = completeness

        # 2. Validité
        validity = DataQualityScorer._evaluate_validity(df)
        results['dimensions']['validity'] = validity

        # 3. Cohérence
        consistency = DataQualityScorer._evaluate_consistency(df)
        results['dimensions']['consistency'] = consistency

        # 4. Unicité
        uniqueness = DataQualityScorer._evaluate_uniqueness(df)
        results['dimensions']['uniqueness'] = uniqueness

        # 5. Exactitude (heuristique)
        accuracy = DataQualityScorer._evaluate_accuracy(df)
        results['dimensions']['accuracy'] = accuracy

        # Score global (moyenne pondérée)
        weights = {
            'completeness': 0.30,
            'validity': 0.25,
            'consistency': 0.20,
            'uniqueness': 0.15,
            'accuracy': 0.10
        }

        overall = sum(
            results['dimensions'][dim]['score'] * weights[dim]
            for dim in weights.keys()
        )

        results['overall_score'] = round(overall, 2)
        results['grade'] = DataQualityScorer._get_grade(overall)

        # Générer les recommandations
        results['recommendations'] = DataQualityScorer._generate_recommendations(results['dimensions'])

        # Résumé exécutif
        results['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'usable_rows': DataQualityScorer._count_usable_rows(df),
            'problematic_columns': DataQualityScorer._count_problematic_columns(df),
            'ready_for_ml': overall >= 70
        }

        return results

    @staticmethod
    def _evaluate_completeness(df: pd.DataFrame) -> Dict[str, Any]:
        """Évalue la complétude (absence de valeurs manquantes)"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_rate = ((total_cells - missing_cells) / total_cells) * 100

        # Détails par colonne
        missing_by_column = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            if missing_pct > 0:
                missing_by_column[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }

        return {
            'score': round(completeness_rate, 2),
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells),
            'columns_with_missing': len(missing_by_column),
            'details': missing_by_column,
            'status': 'excellent' if completeness_rate >= 95 else
                     'good' if completeness_rate >= 80 else
                     'fair' if completeness_rate >= 60 else 'poor'
        }

    @staticmethod
    def _evaluate_validity(df: pd.DataFrame) -> Dict[str, Any]:
        """Évalue la validité (types de données corrects, valeurs dans les plages attendues)"""
        issues = []
        total_checks = 0
        failed_checks = 0

        for col in df.columns:
            total_checks += 1

            # Check 1: Valeurs négatives dans des colonnes qui semblent être des IDs
            if 'id' in col.lower():
                negative_count = (df[col] < 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
                if negative_count > 0:
                    failed_checks += 1
                    issues.append({
                        'column': col,
                        'issue': 'Valeurs négatives dans un ID',
                        'count': int(negative_count)
                    })

            # Check 2: Valeurs infinies dans les colonnes numériques
            if pd.api.types.is_numeric_dtype(df[col]):
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    failed_checks += 1
                    issues.append({
                        'column': col,
                        'issue': 'Valeurs infinies',
                        'count': int(inf_count)
                    })

            # Check 3: Chaînes vides dans les colonnes texte
            if pd.api.types.is_object_dtype(df[col]):
                empty_strings = (df[col].astype(str).str.strip() == '').sum()
                if empty_strings > 0:
                    failed_checks += 1
                    issues.append({
                        'column': col,
                        'issue': 'Chaînes vides',
                        'count': int(empty_strings)
                    })

            # Check 4: Valeurs hors plage pour les pourcentages
            if 'pct' in col.lower() or 'percent' in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
                    if out_of_range > 0:
                        failed_checks += 1
                        issues.append({
                            'column': col,
                            'issue': 'Pourcentage hors plage [0-100]',
                            'count': int(out_of_range)
                        })

        validity_score = ((total_checks - failed_checks) / total_checks * 100) if total_checks > 0 else 100

        return {
            'score': round(validity_score, 2),
            'total_checks': total_checks,
            'failed_checks': failed_checks,
            'issues': issues,
            'status': 'excellent' if validity_score >= 95 else
                     'good' if validity_score >= 80 else
                     'fair' if validity_score >= 60 else 'poor'
        }

    @staticmethod
    def _evaluate_consistency(df: pd.DataFrame) -> Dict[str, Any]:
        """Évalue la cohérence (formats uniformes, pas de contradictions)"""
        issues = []
        total_checks = 0
        failed_checks = 0

        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                total_checks += 1

                # Check: Variations de casse (ex: "Paris", "paris", "PARIS")
                unique_lower = df[col].dropna().astype(str).str.lower().nunique()
                unique_original = df[col].dropna().nunique()

                if unique_lower < unique_original:
                    failed_checks += 1
                    issues.append({
                        'column': col,
                        'issue': 'Variations de casse détectées',
                        'unique_lower': int(unique_lower),
                        'unique_original': int(unique_original)
                    })

                # Check: Espaces en début/fin
                total_checks += 1
                trimmed = df[col].dropna().astype(str).str.strip()
                original = df[col].dropna().astype(str)
                whitespace_issues = (trimmed != original).sum()

                if whitespace_issues > 0:
                    failed_checks += 1
                    issues.append({
                        'column': col,
                        'issue': 'Espaces en début/fin',
                        'count': int(whitespace_issues)
                    })

        # Check global: colonnes avec variance nulle (toutes les valeurs identiques)
        for col in df.select_dtypes(include=[np.number]).columns:
            total_checks += 1
            if df[col].nunique() == 1:
                failed_checks += 1
                issues.append({
                    'column': col,
                    'issue': 'Colonne constante (aucune variance)',
                    'value': str(df[col].iloc[0])
                })

        consistency_score = ((total_checks - failed_checks) / total_checks * 100) if total_checks > 0 else 100

        return {
            'score': round(consistency_score, 2),
            'total_checks': total_checks,
            'failed_checks': failed_checks,
            'issues': issues,
            'status': 'excellent' if consistency_score >= 95 else
                     'good' if consistency_score >= 80 else
                     'fair' if consistency_score >= 60 else 'poor'
        }

    @staticmethod
    def _evaluate_uniqueness(df: pd.DataFrame) -> Dict[str, Any]:
        """Évalue l'unicité (détection de doublons)"""
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        uniqueness_rate = ((total_rows - duplicate_rows) / total_rows * 100) if total_rows > 0 else 100

        # Colonnes avec trop de doublons
        low_cardinality_cols = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.01 and df[col].nunique() > 1:  # Moins de 1% de valeurs uniques
                low_cardinality_cols.append({
                    'column': col,
                    'unique_count': int(df[col].nunique()),
                    'unique_ratio': round(unique_ratio * 100, 2)
                })

        return {
            'score': round(uniqueness_rate, 2),
            'duplicate_rows': int(duplicate_rows),
            'total_rows': total_rows,
            'low_cardinality_columns': low_cardinality_cols,
            'status': 'excellent' if uniqueness_rate >= 98 else
                     'good' if uniqueness_rate >= 90 else
                     'fair' if uniqueness_rate >= 70 else 'poor'
        }

    @staticmethod
    def _evaluate_accuracy(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Évalue l'exactitude basée sur des heuristiques
        (impossible de vérifier sans source de vérité externe)
        """
        issues = []
        total_checks = 0
        failed_checks = 0

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                total_checks += 1

                # Détection d'outliers (méthode IQR)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()

                if outliers > len(df) * 0.05:  # Plus de 5% d'outliers
                    failed_checks += 1
                    issues.append({
                        'column': col,
                        'issue': 'Outliers suspects (>5%)',
                        'count': int(outliers),
                        'percentage': round(outliers / len(df) * 100, 2)
                    })

        accuracy_score = ((total_checks - failed_checks) / total_checks * 100) if total_checks > 0 else 100

        return {
            'score': round(accuracy_score, 2),
            'total_checks': total_checks,
            'failed_checks': failed_checks,
            'issues': issues,
            'status': 'excellent' if accuracy_score >= 95 else
                     'good' if accuracy_score >= 80 else
                     'fair' if accuracy_score >= 60 else 'poor'
        }

    @staticmethod
    def _get_grade(score: float) -> str:
        """Convertit un score en grade"""
        if score >= 90:
            return "A - Excellent"
        elif score >= 80:
            return "B - Bon"
        elif score >= 70:
            return "C - Acceptable"
        elif score >= 60:
            return "D - Médiocre"
        else:
            return "F - Insuffisant"

    @staticmethod
    def _generate_recommendations(dimensions: Dict[str, Dict]) -> List[str]:
        """Génère des recommandations basées sur les scores"""
        recommendations = []

        # Complétude
        comp = dimensions['completeness']
        if comp['score'] < 95:
            if comp['columns_with_missing'] > 0:
                recommendations.append(
                    f"[CRITIQUE] **Complétude**: {comp['columns_with_missing']} colonnes ont des valeurs manquantes. "
                    f"Considérez l'imputation ou la suppression de ces colonnes."
                )

        # Validité
        val = dimensions['validity']
        if val['failed_checks'] > 0:
            recommendations.append(
                f"[CRITIQUE] **Validité**: {val['failed_checks']} problèmes de validité détectés. "
                f"Corrigez les valeurs invalides avant l'analyse."
            )

        # Cohérence
        cons = dimensions['consistency']
        if cons['failed_checks'] > 0:
            recommendations.append(
                f"[ATTENTION] **Cohérence**: {cons['failed_checks']} incohérences trouvées. "
                f"Standardisez les formats (casse, espaces) pour améliorer la qualité."
            )

        # Unicité
        uniq = dimensions['uniqueness']
        if uniq['duplicate_rows'] > 0:
            recommendations.append(
                f"[ATTENTION] **Unicité**: {uniq['duplicate_rows']} lignes dupliquées détectées. "
                f"Vérifiez et supprimez les doublons si nécessaire."
            )

        # Exactitude
        acc = dimensions['accuracy']
        if acc['failed_checks'] > 0:
            recommendations.append(
                f"[ATTENTION] **Exactitude**: {acc['failed_checks']} colonnes avec outliers suspects. "
                f"Vérifiez ces valeurs extrêmes avant l'entraînement."
            )

        if not recommendations:
            recommendations.append("**Excellent** - Aucune recommandation majeure. Vos données sont de haute qualité.")

        return recommendations

    @staticmethod
    def _count_usable_rows(df: pd.DataFrame) -> int:
        """Compte les lignes sans valeurs manquantes"""
        return int(df.dropna().shape[0])

    @staticmethod
    def _count_problematic_columns(df: pd.DataFrame) -> int:
        """Compte les colonnes avec plus de 30% de valeurs manquantes"""
        threshold = 0.3
        problematic = sum(
            df[col].isnull().sum() / len(df) > threshold
            for col in df.columns
        )
        return problematic

    @staticmethod
    def format_report(quality_results: Dict[str, Any]) -> str:
        """Formate le rapport de qualité en Markdown pour affichage"""
        report = "# Rapport de Qualité des Données\n\n"

        # Score global
        report += f"## Score Global: **{quality_results['overall_score']}/100** - {quality_results['grade']}\n\n"

        # Indicateur visuel
        score = quality_results['overall_score']
        bar_length = int(score / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        report += f"`{bar}` {score}%\n\n"

        # Résumé
        summary = quality_results['summary']
        report += "## Résumé\n\n"
        report += f"- **Lignes totales:** {summary['total_rows']:,}\n"
        report += f"- **Colonnes totales:** {summary['total_columns']}\n"
        report += f"- **Lignes exploitables (sans NaN):** {summary['usable_rows']:,}\n"
        report += f"- **Colonnes problématiques:** {summary['problematic_columns']}\n"
        report += f"- **Prêt pour le ML:** {'Oui' if summary['ready_for_ml'] else 'Non'}\n\n"

        # Dimensions
        report += "## Évaluation par Dimension\n\n"

        dimension_names = {
            'completeness': '1. Complétude',
            'validity': '2. Validité',
            'consistency': '3. Cohérence',
            'uniqueness': '4. Unicité',
            'accuracy': '5. Exactitude'
        }

        for key, name in dimension_names.items():
            dim = quality_results['dimensions'][key]

            report += f"### {name} - {dim['score']}/100 ({dim['status'].upper()})\n\n"

            # Détails spécifiques
            if key == 'completeness' and dim['columns_with_missing'] > 0:
                report += f"- Valeurs manquantes: {dim['missing_cells']:,} cellules\n"
                report += f"- Colonnes affectées: {dim['columns_with_missing']}\n"

            if key == 'validity' and dim['failed_checks'] > 0:
                report += f"- Checks échoués: {dim['failed_checks']}/{dim['total_checks']}\n"

            if key == 'uniqueness' and dim['duplicate_rows'] > 0:
                report += f"- Lignes dupliquées: {dim['duplicate_rows']}\n"

            report += "\n"

        # Recommandations
        report += "## Recommandations\n\n"
        for rec in quality_results['recommendations']:
            report += f"{rec}\n\n"

        return report
