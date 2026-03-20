"""Page d'exploration"""
import streamlit as st
import base64
from backend.utils.data_quality_scorer import DataQualityScorer
from frontend.utils.ui_helpers import page_header, section_title


def render_exploration():
    if not st.session_state.dataset_loaded:
        st.warning("**Étape précédente requise**")
        st.info("Veuillez d'abord charger un fichier depuis la page d'accueil.")
        if st.button("Retour à l'accueil", type="primary"):
            st.session_state.current_page = "home"
            st.rerun()
        return

    page_header("🔍", "Exploration des données",
                "Analysez la qualité et comprenez la structure de votre dataset",
                badge="Étape 2/5")

    if 'workflow' in st.session_state:
        st.session_state.workflow['step_2_data_explored'] = True

    tab1, tab2 = st.tabs(["Vue Générale", "Analyse EDA"])

    with tab1:
        show_overview_tab()

    with tab2:
        show_eda_complet_tab()

    st.markdown("<br>", unsafe_allow_html=True)
    quality_score = st.session_state.dataset_info.get('quality_score', 0)
    q_color = "#22c55e" if quality_score >= 70 else "#f59e0b"

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{q_color}10,{q_color}05);'
        f'border:1px solid {q_color}30;border-left:4px solid {q_color};'
        f'border-radius:14px;padding:20px 24px;margin-bottom:16px;'
        f'display:flex;align-items:center;justify-content:space-between;gap:16px;">'
        f'<div>'
        f'<div style="font-weight:700;color:#0f172a;font-size:15px;margin-bottom:4px;">'
        f'Prêt pour la modélisation'
        f'</div>'
        f'<div style="font-size:13px;color:#64748b;">'
        f'Score qualité : <strong style="color:{q_color};">{quality_score}/100</strong>'
        f'{"— données exploitables" if quality_score >= 70 else "— amélioration recommandée"}'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        label = "🧠 Passer à la modélisation" if quality_score >= 70 else "🧠 Continuer malgré tout"
        if st.button(label, type="primary", use_container_width=True):
            st.session_state.current_page = "predict"
            st.rerun()


def show_overview_tab():
    st.markdown("### Informations du Dataset")

    info = st.session_state.dataset_info
    profile = st.session_state.dataset_profile

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", f"{info['rows']:,}")
    with col2:
        st.metric("Colonnes", info['columns'])
    with col3:
        missing = profile['missing_values']['total']
        st.metric("Valeurs manquantes", f"{missing:,}")

    st.markdown("### Repartition des Types de Colonnes")
    col_types = profile['column_types']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numeriques", col_types['numeric'])
    with col2:
        st.metric("Categorielles", col_types['categorical'])
    with col3:
        st.metric("Dates", col_types['datetime'])

    st.markdown("### Details des Colonnes")
    
    column_data = []
    for col in info['column_names']:
        dtype = info['dtypes'][col]
        column_data.append({
            'Nom': col,
            'Type': dtype
        })
    
    st.dataframe(column_data, use_container_width=True)

    if profile['missing_values']['total'] > 0 or profile['duplicates'] > 0:
        st.markdown("### Qualite des Donnees")
        
        if profile['missing_values']['total'] > 0:
            st.warning(f"Attention: {profile['missing_values']['total']} valeurs manquantes detectees ({profile['missing_values']['percentage']:.1f}%)")
        
        if profile['duplicates'] > 0:
            st.warning(f"Attention: {profile['duplicates']} lignes dupliquees")


def show_quality_report_tab():
    st.markdown("## Rapport de Qualité des Données")
    st.markdown("Évaluation professionnelle selon les standards de l'industrie")

    profile = st.session_state.dataset_profile

    if 'quality_report' not in profile:
        st.info("Rapport de qualité non disponible. Rechargez le fichier pour générer le rapport.")
        return

    quality_report = profile['quality_report']

    st.markdown("### Score Global")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        score = quality_report['overall_score']
        grade = quality_report['grade']

        if score >= 90:
            color = "#28a745"  # Vert
        elif score >= 70:
            color = "#ffc107"  # Jaune
        elif score >= 50:
            color = "#fd7e14"  # Orange
        else:
            color = "#dc3545"

        st.markdown(
            f'<div style="background-color:#e0e0e0;border-radius:10px;height:40px;overflow:hidden;">'
            f'<div style="background-color:{color};width:{score}%;height:100%;display:flex;'
            f'align-items:center;justify-content:center;color:white;font-weight:bold;font-size:18px;">'
            f'{score}/100'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.metric("Note", grade.split(' - ')[0], delta=grade.split(' - ')[1])

    with col3:
        ready_ml = quality_report['summary']['ready_for_ml']
        st.metric("ML Ready", "Oui" if ready_ml else "Non")

    st.markdown("---")

    st.markdown("### Résumé Exécutif")

    summary = quality_report['summary']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lignes Totales", f"{summary['total_rows']:,}")
    with col2:
        st.metric("Colonnes", summary['total_columns'])
    with col3:
        usable_pct = (summary['usable_rows'] / summary['total_rows'] * 100) if summary['total_rows'] > 0 else 0
        st.metric("Lignes Utilisables", f"{summary['usable_rows']:,}", delta=f"{usable_pct:.1f}%")
    with col4:
        st.metric("Colonnes Problématiques", summary['problematic_columns'])

    st.markdown("---")

    st.markdown("### Évaluation par Dimension")

    dimensions = quality_report['dimensions']

    dimension_names = {
        'completeness': ('Complétude', 'Absence de valeurs manquantes'),
        'validity': ('Validité', 'Types et plages de valeurs corrects'),
        'consistency': ('Cohérence', 'Formats uniformes et absence de contradictions'),
        'uniqueness': ('Unicité', 'Absence de doublons'),
        'accuracy': ('Exactitude', 'Détection heuristique des anomalies')
    }

    for key, (name, description) in dimension_names.items():
        dim = dimensions[key]

        with st.expander(f"{name} - {dim['score']}/100 ({dim['status'].upper()})", expanded=(dim['status'] in ['fair', 'poor'])):
            st.markdown(f"**Description:** {description}")

            score_color = "#28a745" if dim['status'] == 'excellent' else \
                         "#ffc107" if dim['status'] == 'good' else \
                         "#fd7e14" if dim['status'] == 'fair' else "#dc3545"

            st.markdown(
                f'<div style="background-color:#f0f0f0;border-radius:5px;padding:10px;margin:10px 0;">'
                f'<div style="background-color:{score_color};width:{dim["score"]}%;height:20px;border-radius:3px;"></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            if key == 'completeness':
                if dim['columns_with_missing'] > 0:
                    st.warning(f" {dim['columns_with_missing']} colonnes avec valeurs manquantes ({dim['missing_cells']:,} cellules)")

                    if dim['details']:
                        st.markdown("**Top 5 colonnes affectées:**")
                        sorted_missing = sorted(dim['details'].items(), key=lambda x: x[1]['percentage'], reverse=True)[:5]
                        for col, info in sorted_missing:
                            st.markdown(f"- `{col}`: {info['count']} valeurs ({info['percentage']:.1f}%)")

            elif key == 'validity':
                if dim['issues']:
                    st.error(f"{len(dim['issues'])} problèmes de validité détectés")
                    for issue in dim['issues'][:5]:
                        st.markdown(f"- **{issue['column']}**: {issue['issue']} ({issue['count']} occurrences)")

            elif key == 'consistency':
                if dim['issues']:
                    st.warning(f"{len(dim['issues'])} incohérences trouvées")
                    for issue in dim['issues'][:5]:
                        st.markdown(f"- **{issue['column']}**: {issue['issue']}")

            elif key == 'uniqueness':
                if dim['duplicate_rows'] > 0:
                    dup_pct = (dim['duplicate_rows'] / dim['total_rows'] * 100)
                    st.warning(f"{dim['duplicate_rows']} lignes dupliquées ({dup_pct:.2f}%)")

                if dim['low_cardinality_columns']:
                    st.info(f"{len(dim['low_cardinality_columns'])} colonnes à faible cardinalité")

            elif key == 'accuracy':
                if dim['issues']:
                    st.warning(f"Outliers détectés dans {len(dim['issues'])} colonnes")
                    for issue in dim['issues'][:5]:
                        st.markdown(f"- **{issue['column']}**: {issue['count']} outliers ({issue['percentage']:.1f}%)")

    st.markdown("---")

    st.markdown("### Recommandations")

    for rec in quality_report['recommendations']:
        if "CRITIQUE" in rec.upper() or "CRITICAL" in rec.upper():
            st.error(rec)
        elif "ATTENTION" in rec.upper() or "WARNING" in rec.upper():
            st.warning(rec)
        else:
            st.success(rec)

    st.markdown("---")

    st.markdown("### Export du Rapport")

    col1, col2 = st.columns(2)

    with col1:
        markdown_report = DataQualityScorer.format_report(quality_report)

        st.download_button(
            label="Télécharger le rapport (Markdown)",
            data=markdown_report,
            file_name="rapport_qualite_donnees.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        import json
        import numpy as np

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

        quality_report_clean = convert_numpy_types(quality_report)
        json_report = json.dumps(quality_report_clean, indent=2, ensure_ascii=False)

        st.download_button(
            label=" Télécharger les données (JSON)",
            data=json_report,
            file_name="rapport_qualite_donnees.json",
            mime="application/json",
            use_container_width=True
        )


def show_eda_complet_tab():
    st.markdown("### Analyse Exploratoire des Donnees (EDA)")

    st.info("""
    **Cette analyse complète examine tous les aspects de vos données:**

     Aperçu général et types de variables
     Qualité des données (valeurs manquantes, doublons, cardinalité)
     Statistiques descriptives détaillées (moyenne, médiane, asymétrie, kurtosis)
     Distribution des variables numériques et catégorielles
     Détection des valeurs aberrantes (outliers)
     Analyse des corrélations (fortes et modérées)
     Recommandations actionnables pour le preprocessing
     Visualisations: histogrammes, boxplots, barplots, heatmap de corrélation
    """)

    st.markdown("---")

    if 'eda_report' in st.session_state and st.session_state.eda_report:
        st.markdown(st.session_state.eda_report)

        if 'eda_visualizations' in st.session_state and st.session_state.eda_visualizations:
            visualizations = st.session_state.eda_visualizations

            st.markdown("---")
            st.markdown("## VISUALISATIONS")

            if 'histogrammes' in visualizations:
                st.markdown("### Distributions des variables numeriques (Histogrammes)")
                img_data = base64.b64decode(visualizations['histogrammes'])
                st.image(img_data, use_column_width=True)
                st.markdown("---")

            if 'boxplots' in visualizations:
                st.markdown("### Detection des outliers (Boxplots)")
                img_data = base64.b64decode(visualizations['boxplots'])
                st.image(img_data, use_column_width=True)
                st.markdown("---")

            if 'barplots' in visualizations:
                st.markdown("### Repartition des variables categorielles (Top valeurs)")
                img_data = base64.b64decode(visualizations['barplots'])
                st.image(img_data, use_column_width=True)
                st.markdown("---")

            if 'correlation_heatmap' in visualizations:
                st.markdown("### Matrice de Correlation")
                img_data = base64.b64decode(visualizations['correlation_heatmap'])
                st.image(img_data, use_column_width=True)
                st.markdown("---")

        st.markdown("---")

    show_preprocessing_plan()

    eda_already_done = bool(st.session_state.get('eda_report'))
    btn_label = "🔄 Relancer l'EDA" if eda_already_done else "🔍 Lancer l'EDA Complet"
    btn_help  = "Regénère le rapport et met à jour le plan de preprocessing" if eda_already_done else None

    if st.button(btn_label, type="primary" if not eda_already_done else "secondary", help=btn_help):
        dataset = st.session_state.get('shared_dataset')
        if dataset is None:
            st.error("Aucun dataset chargé. Veuillez d'abord charger un fichier.")
            return

        with st.spinner("Analyse en cours... Cela peut prendre quelques instants..."):
            try:
                from backend.agents.analyste_agent import AnalysteAgent
                from backend.utils.preprocessing_advisor import build_preprocessing_plan
                agent = AnalysteAgent()
                result, visualizations = agent._eda_complet(dataset)
                st.session_state.eda_report = result
                st.session_state.eda_visualizations = visualizations
                st.session_state.preprocessing_plan = build_preprocessing_plan(dataset)
                # Réinitialise le dataset nettoyé si on relance l'EDA
                st.session_state.pop('clean_dataset', None)
                st.session_state.pop('preprocessing_rapport', None)
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de l'analyse EDA : {e}")
                import traceback
                st.code(traceback.format_exc())


def show_preprocessing_plan():
    """Affiche le plan de preprocessing expert et permet de préparer le dataset."""
    plan = st.session_state.get('preprocessing_plan')
    if not plan:
        return

    st.markdown("---")
    st.markdown("## Plan de Preprocessing Expert")
    st.markdown("*Décisions basées sur l'analyse statistique réelle du dataset*")

    summary = plan.get('summary', {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Colonnes analysées", summary.get('total_columns', 0))
    with col2:
        st.metric("Colonnes avec NaN", summary.get('cols_with_missing', 0))
    with col3:
        st.metric("Colonnes avec outliers", summary.get('cols_with_outliers', 0))
    with col4:
        st.metric("À supprimer", summary.get('cols_to_drop', 0))

    # Détail par colonne
    cols_plan = plan.get('columns', {})

    # Colonnes avec actions
    treated_cols = {col: info for col, info in cols_plan.items()
                    if info['action_missing'] != 'none' or info['action_outliers'] == 'cap' or info['drop']}

    if treated_cols:
        with st.expander(f"Détail des décisions ({len(treated_cols)} colonnes)", expanded=True):
            for col, info in treated_cols.items():
                status_color = "#ef4444" if info['drop'] else "#f59e0b" if info['action_missing'] != 'none' else "#22c55e"
                part_drop = (
                    '<br><span style="font-size:12px;color:#ef4444;">🗑 ' + info["drop_reason"] + '</span>'
                    if info["drop"] else ""
                )
                part_nan = (
                    '<br><span style="font-size:12px;color:#f59e0b;">🔧 NaN: ' + info["reason_missing"] + ' → ' + info["action_missing"] + '</span>'
                    if info["action_missing"] != "none" and not info["drop"] else ""
                )
                part_outliers = (
                    '<br><span style="font-size:12px;color:#8b5cf6;">📊 Outliers: ' + info["reason_outliers"] + '</span>'
                    if info["action_outliers"] == "cap" else ""
                )
                st.markdown(
                    f'<div style="border-left:3px solid {status_color};padding:8px 14px;'
                    f'margin-bottom:6px;background:#f8fafc;border-radius:0 6px 6px 0;">'
                    f'<strong style="color:#0f172a;">{col}</strong> '
                    f'<span style="font-size:11px;color:#64748b;">({info["dtype"]})</span>'
                    f'{part_drop}{part_nan}{part_outliers}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Corrélations détectées
    corr_pairs = plan.get('global', {}).get('correlated_pairs', [])
    if corr_pairs:
        with st.expander(f"Multicolinéarité détectée ({len(corr_pairs)} paires)"):
            for col_drop, col_keep, corr_val in corr_pairs:
                st.markdown(f"- **{col_drop}** ↔ **{col_keep}** : r={corr_val:.3f} → `{col_drop}` sera supprimée")

    # ── Statut actuel ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    plan_applied = 'clean_dataset' in st.session_state

    if plan_applied:
        rapport_prec = st.session_state.get('preprocessing_rapport', {})
        df_clean = st.session_state.clean_dataset
        dataset  = st.session_state.get('shared_dataset')
        st.markdown(
            '<div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.4);'
            'border-left:4px solid #22c55e;border-radius:8px;padding:12px 16px;margin-bottom:12px;">'
            '<strong style="color:#166534;">✅ Plan appliqué — dataset prêt pour la modélisation</strong>'
            '</div>',
            unsafe_allow_html=True
        )

        # Tableau avant / après
        import pandas as _pd
        rows = []
        for col, action in rapport_prec.get('actions', {}).items():
            rows.append({'Colonne': col, 'Action appliquée': action})
        if rows:
            with st.expander("📋 Détail des modifications appliquées", expanded=False):
                st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            delta_rows = rapport_prec.get('rows_after', 0) - rapport_prec.get('rows_before', 0)
            st.metric("Lignes", f"{rapport_prec.get('rows_after', 0):,}",
                      delta=f"{delta_rows:+,}" if delta_rows else None)
        with c2:
            n_before = dataset.shape[1] if dataset is not None else '?'
            n_after  = df_clean.shape[1]
            st.metric("Colonnes", n_after,
                      delta=f"{n_after - n_before:+}" if dataset is not None else None)
        with c3:
            nan_after = int(df_clean.isnull().sum().sum())
            st.metric("NaN restants", nan_after,
                      delta="0 NaN" if nan_after == 0 else f"{nan_after} restants")

        if st.button("↩️ Réinitialiser le preprocessing", type="secondary"):
            st.session_state.pop('clean_dataset', None)
            st.session_state.pop('preprocessing_rapport', None)
            st.rerun()
    else:
        st.markdown(
            '<div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.4);'
            'border-left:4px solid #f59e0b;border-radius:8px;padding:12px 16px;margin-bottom:12px;">'
            '<strong style="color:#92400e;">⚠️ Plan non encore appliqué</strong>'
            '<br><span style="font-size:13px;color:#78350f;">Le dataset brut sera utilisé pour la modélisation. '
            'Cliquez ci-dessous pour appliquer les recommandations.</span>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("🧹 Appliquer le plan et préparer le dataset", type="primary"):
            dataset = st.session_state.get('shared_dataset')
            if dataset is None:
                st.error("Dataset non disponible.")
                return
            with st.spinner("Application du plan de preprocessing..."):
                try:
                    from backend.utils.data_preprocessor import DataPreprocessor
                    df_clean, rapport = DataPreprocessor.apply_plan(dataset, plan)
                    st.session_state.clean_dataset = df_clean
                    st.session_state.preprocessing_rapport = rapport
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    import traceback
                    st.code(traceback.format_exc())
