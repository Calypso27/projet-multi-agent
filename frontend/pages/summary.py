"""Page Executive Summary - Rapport synthétique pour le client"""
import streamlit as st
from datetime import datetime
import pandas as pd
from backend.models.model_manager import ModelManager
from backend.utils.data_quality_scorer import DataQualityScorer


def render_summary():
    """Affiche le résumé exécutif professionnel"""

    st.markdown("# Executive Summary")
    st.markdown("*Rapport synthétique pour la présentation client*")

    # Vérifier les prérequis
    if not st.session_state.get('dataset_loaded'):
        st.warning("**Étape précédente requise**")
        st.info("Commencez par charger un dataset pour générer un rapport.")
        if st.button("Aller à l'accueil", type="primary"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return

    workflow = st.session_state.get('workflow', {})
    if not workflow.get('step_2_data_explored', False):
        st.warning("**Étape précédente requise**")
        st.info("Explorez d'abord vos données pour avoir un rapport complet.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Explorer les données", type="primary", use_container_width=True):
                st.session_state.current_page = "explore"
                st.session_state.current_page_display = "Explorer"
                st.rerun()
        with col2:
            if st.button("Générer le rapport quand même", use_container_width=True):
                st.session_state.workflow['step_2_data_explored'] = True
                st.rerun()
        return

    # En-tête professionnel
    st.markdown("---")

    # Informations du projet
    info = st.session_state.dataset_info
    profile = st.session_state.dataset_profile

    # Section 1: Vue d'ensemble du projet
    st.markdown("## 1. Vue d'ensemble du projet")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        **Dataset analysé:** `{info['filename']}`
        **Date d'analyse:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
        **Volume de données:** {info['rows']:,} lignes × {info['columns']} colonnes
        **Format:** {info['format'].upper()}
        **Type de données:** {info['data_type']}
        """)

    with col2:
        # Score de qualité global
        quality_score = info.get('quality_score', 0)
        quality_grade = info.get('quality_grade', 'N/A')

        if quality_score >= 90:
            color = "green"
            status = "Excellente"
        elif quality_score >= 70:
            color = "orange"
            status = "Bonne"
        else:
            color = "red"
            status = "À améliorer"

        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center;">
            <h3 style="margin: 0; color: {color};">Qualité des données</h3>
            <h1 style="margin: 10px 0; color: {color};">{quality_score}/100</h1>
            <p style="margin: 0; font-weight: bold;">{quality_grade}</p>
            <p style="margin: 5px 0; font-size: 14px;">{status}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Indicateurs clés (KPI)
    st.markdown("## 2. Indicateurs clés de performance (KPI)")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        completeness = 100 - (profile['missing_values']['percentage'] if profile['missing_values']['total'] > 0 else 0)
        st.metric(
            "Complétude des données",
            f"{completeness:.1f}%",
            delta="Excellent" if completeness >= 95 else "À améliorer"
        )

    with kpi_col2:
        duplicates_pct = (profile['duplicates'] / info['rows'] * 100) if info['rows'] > 0 else 0
        st.metric(
            "Données uniques",
            f"{100 - duplicates_pct:.1f}%",
            delta=f"{profile['duplicates']} doublons" if profile['duplicates'] > 0 else "Aucun doublon"
        )

    with kpi_col3:
        quality_report = profile.get('quality_report', {})
        usable_rows = quality_report.get('summary', {}).get('usable_rows', info['rows'])
        usable_pct = (usable_rows / info['rows'] * 100) if info['rows'] > 0 else 0
        st.metric(
            "Lignes exploitables",
            f"{usable_rows:,}",
            delta=f"{usable_pct:.1f}%"
        )

    with kpi_col4:
        # Compter les modèles entraînés
        models = ModelManager.list_models()
        models_for_dataset = [m for m in models if m['target'] in info.get('column_names', [])]
        st.metric(
            "Modèles ML entraînés",
            len(models_for_dataset),
            delta="Prêt" if len(models_for_dataset) > 0 else "En attente"
        )

    st.markdown("---")

    # Section 3: Analyse de la qualité des données
    st.markdown("## 3. Évaluation de la qualité des données")

    if 'quality_report' in profile:
        quality_report = profile['quality_report']
        dimensions = quality_report['dimensions']

        # Graphique en barres des dimensions
        dim_names = ['Complétude', 'Validité', 'Cohérence', 'Unicité', 'Exactitude']
        dim_keys = ['completeness', 'validity', 'consistency', 'uniqueness', 'accuracy']
        dim_scores = [dimensions[key]['score'] for key in dim_keys]

        # Créer un DataFrame pour l'affichage
        quality_df = pd.DataFrame({
            'Dimension': dim_names,
            'Score': dim_scores,
            'Statut': [dimensions[key]['status'].upper() for key in dim_keys]
        })

        st.dataframe(quality_df, use_container_width=True, hide_index=True)

        # Recommandations prioritaires
        st.markdown("### Recommandations prioritaires")

        recommendations = quality_report.get('recommendations', [])
        critical_recs = [r for r in recommendations if "CRITIQUE" in r.upper() or "CRITICAL" in r.upper()]
        warning_recs = [r for r in recommendations if "ATTENTION" in r.upper() or "WARNING" in r.upper()]

        if critical_recs:
            st.error("**Actions critiques requises:**")
            for rec in critical_recs:
                st.markdown(f"- {rec}")

        if warning_recs:
            st.warning("**Améliorations recommandées:**")
            for rec in warning_recs[:3]:  # Top 3
                st.markdown(f"- {rec}")

        if not critical_recs and not warning_recs:
            st.success("Aucune action requise. Les données sont de qualité optimale.")

    st.markdown("---")

    # Section 4: Caractéristiques des données
    st.markdown("## 4. Caractéristiques des données")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Distribution des types de colonnes")

        col_types = profile['column_types']
        type_data = {
            'Numériques': col_types['numeric'],
            'Catégorielles': col_types['categorical'],
            'Dates': col_types['datetime']
        }

        type_df = pd.DataFrame([
            {'Type': k, 'Nombre': v, 'Pourcentage': f"{v/info['columns']*100:.1f}%"}
            for k, v in type_data.items() if v > 0
        ])

        st.dataframe(type_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Points d'attention")

        attention_points = []

        if profile['missing_values']['total'] > 0:
            attention_points.append(
                f"- {profile['missing_values']['total']:,} valeurs manquantes "
                f"({profile['missing_values']['percentage']:.1f}%)"
            )

        if profile['duplicates'] > 0:
            attention_points.append(f"- {profile['duplicates']} lignes dupliquées")

        if 'quality_report' in profile:
            problematic_cols = profile['quality_report']['summary'].get('problematic_columns', 0)
            if problematic_cols > 0:
                attention_points.append(f"- {problematic_cols} colonnes problématiques")

        if attention_points:
            for point in attention_points:
                st.markdown(point)
        else:
            st.success("Aucun point d'attention majeur")

    st.markdown("---")

    # Section 5: Modèles Machine Learning
    st.markdown("## 5. Modèles Machine Learning développés")

    models = ModelManager.list_models()

    if models:
        st.success(f"{len(models)} modèle(s) entraîné(s) et disponible(s) pour déploiement")

        # Tableau des modèles
        models_df = pd.DataFrame([{
            'Modèle': m['name'],
            'Type': m['type'].capitalize(),
            'Variable cible': m['target'],
            'Performance': _format_model_performance(m['metrics'], m['type']),
            'Date création': pd.to_datetime(m['created_at']).strftime('%d/%m/%Y %H:%M'),
            'Features': m['n_features']
        } for m in models])

        st.dataframe(models_df, use_container_width=True, hide_index=True)

        # Meilleur modèle
        if models:
            best_model = models[0]  # Les modèles sont triés par date
            st.markdown(f"""
            **Modèle recommandé pour le déploiement:**
            - **Nom:** {best_model['name']}
            - **Performance:** {_format_model_performance(best_model['metrics'], best_model['type'])}
            - **Prédiction de:** {best_model['target']}
            """)

    else:
        st.info("Aucun modèle ML n'a encore été entraîné. Rendez-vous dans l'onglet **Modélisation** pour créer un modèle.")

        if st.button("Entraîner un modèle maintenant", type="primary"):
            st.session_state.current_page = "predict"
            st.session_state.current_page_display = "Prédire"
            st.rerun()

    st.markdown("---")

    # Section 6: Prochaines étapes
    st.markdown("## 6. Prochaines étapes recommandées")

    next_steps = []

    # Vérifier ce qui a été fait
    has_models = len(models) > 0
    quality_ok = info.get('quality_score', 0) >= 70

    if not quality_ok:
        next_steps.append({
            'priorité': 'CRITIQUE',
            'action': 'Améliorer la qualité des données',
            'description': 'Traiter les valeurs manquantes et les incohérences détectées'
        })

    if not has_models:
        next_steps.append({
            'priorité': 'CRITIQUE',
            'action': 'Entraîner un modèle ML',
            'description': 'Créer un modèle prédictif pour exploiter les données'
        })

    if has_models and quality_ok:
        next_steps.append({
            'priorité': 'RECOMMANDÉ',
            'action': 'Déployer le modèle en production',
            'description': 'Utiliser le modèle pour faire des prédictions sur de nouvelles données'
        })

    next_steps.append({
        'priorité': 'OPTIONNEL',
        'action': 'Approfondir l\'analyse exploratoire',
        'description': 'Explorer les corrélations et patterns dans les données'
    })

    next_steps.append({
        'priorité': 'OPTIONNEL',
        'action': 'Optimiser les hyperparamètres',
        'description': 'Améliorer les performances des modèles existants'
    })

    # Afficher sous forme de tableau
    steps_df = pd.DataFrame(next_steps)
    st.dataframe(steps_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 7: Export du rapport
    st.markdown("## 7. Export du rapport")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Générer le rapport complet en markdown
        full_report = _generate_full_markdown_report(info, profile, models)

        st.download_button(
            label="Rapport complet (Markdown)",
            data=full_report,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        # Export des données de qualité
        if 'quality_report' in profile:
            quality_md = DataQualityScorer.format_report(profile['quality_report'])

            st.download_button(
                label="Rapport qualité détaillé",
                data=quality_md,
                file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    with col3:
        # Export JSON pour intégration
        import json
        import numpy as np

        # Fonction pour convertir les types numpy en types Python natifs
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        export_data = {
            'dataset_info': convert_numpy_types(info),
            'quality_score': float(info.get('quality_score', 0)) if info.get('quality_score') else None,
            'models_count': len(models),
            'generated_at': datetime.now().isoformat()
        }

        st.download_button(
            label="Données JSON (API)",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"summary_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")

    # Footer professionnel
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin-top: 30px;">
        <p style="margin: 0; color: #666; font-size: 14px;">
            Rapport généré automatiquement par la Plateforme d'Analyse de Données Multi-Agent
        </p>
        <p style="margin: 5px 0; color: #999; font-size: 12px;">
            © 2026 - Tous droits réservés
        </p>
    </div>
    """, unsafe_allow_html=True)


def _format_model_performance(metrics: dict, problem_type: str) -> str:
    """Formate les métriques de performance du modèle"""
    if problem_type == 'regression':
        r2 = metrics.get('R²', 0)
        return f"R² = {r2:.3f}"
    else:
        acc = metrics.get('Accuracy', 0)
        return f"Accuracy = {acc:.3f}"


def _generate_full_markdown_report(info: dict, profile: dict, models: list) -> str:
    """Génère un rapport markdown complet prêt pour conversion PDF"""

    report = f"""# Executive Summary - Analyse de Données

**Dataset:** {info['filename']}
**Date:** {datetime.now().strftime('%d/%m/%Y %H:%M')}

---

## 1. Vue d'ensemble

- **Volume:** {info['rows']:,} lignes × {info['columns']} colonnes
- **Format:** {info['format'].upper()}
- **Type:** {info['data_type']}
- **Qualité globale:** {info.get('quality_score', 'N/A')}/100 ({info.get('quality_grade', 'N/A')})

## 2. Indicateurs clés

| Métrique | Valeur |
|----------|--------|
| Complétude | {100 - profile['missing_values']['percentage']:.1f}% |
| Valeurs manquantes | {profile['missing_values']['total']:,} |
| Lignes dupliquées | {profile['duplicates']} |
| Colonnes numériques | {profile['column_types']['numeric']} |
| Colonnes catégorielles | {profile['column_types']['categorical']} |

## 3. Qualité des données

"""

    if 'quality_report' in profile:
        quality_report = profile['quality_report']
        dimensions = quality_report['dimensions']

        report += "| Dimension | Score | Statut |\n"
        report += "|-----------|-------|--------|\n"

        dim_names = {
            'completeness': 'Complétude',
            'validity': 'Validité',
            'consistency': 'Cohérence',
            'uniqueness': 'Unicité',
            'accuracy': 'Exactitude'
        }

        for key, name in dim_names.items():
            dim = dimensions[key]
            report += f"| {name} | {dim['score']}/100 | {dim['status'].upper()} |\n"

        report += "\n### Recommandations\n\n"
        for rec in quality_report.get('recommendations', []):
            report += f"- {rec}\n"

    report += "\n## 4. Modèles Machine Learning\n\n"

    if models:
        report += f"**{len(models)} modèle(s) entraîné(s)**\n\n"
        report += "| Modèle | Type | Variable | Performance |\n"
        report += "|--------|------|----------|-------------|\n"

        for m in models:
            perf = _format_model_performance(m['metrics'], m['type'])
            report += f"| {m['name']} | {m['type']} | {m['target']} | {perf} |\n"
    else:
        report += "*Aucun modèle entraîné pour le moment.*\n"

    report += f"\n---\n\n*Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n"

    return report
