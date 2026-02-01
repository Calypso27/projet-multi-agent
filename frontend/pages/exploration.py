"""Page d'exploration"""
import streamlit as st
import time
import base64
from backend.models.message import Message, MessageType
from backend.utils.data_quality_scorer import DataQualityScorer


def render_exploration():
    if not st.session_state.dataset_loaded:
        st.warning("**Étape précédente requise**")
        st.info("Veuillez d'abord charger un fichier depuis la page d'accueil.")
        if st.button("Retour à l'accueil", type="primary"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return

    st.markdown("# Exploration")
    st.markdown("*Analysez la qualité et comprenez vos données*")

    if 'workflow' in st.session_state:
        st.session_state.workflow['step_2_data_explored'] = True

    tab1, tab2 = st.tabs(["Vue Générale", "Analyse EDA"])

    with tab1:
        show_overview_tab()

    with tab2:
        show_eda_complet_tab()

    st.markdown("---")
    st.markdown("### Étape suivante")

    quality_score = st.session_state.dataset_info.get('quality_score', 0)
    if quality_score >= 70:
        st.success(f"Qualité des données: **{quality_score}/100** - Vous pouvez continuer.")
        if st.button("Passer à la modélisation", type="primary", use_container_width=True):
            st.session_state.current_page = "predict"
            st.session_state.current_page_display = "Prédire"
            st.rerun()
    else:
        st.warning(f"Qualité des données: **{quality_score}/100** - Amélioration recommandée.")
        if st.button("Continuer vers la modélisation", use_container_width=True):
            st.session_state.current_page = "predict"
            st.session_state.current_page_display = "Prédire"
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

        st.markdown(f"""
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 40px; overflow: hidden;">
            <div style="background-color: {color}; width: {score}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px;">
                {score}/100
            </div>
        </div>
        """, unsafe_allow_html=True)

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

            st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 10px; margin: 10px 0;">
                <div style="background-color: {score_color}; width: {dim['score']}%; height: 20px; border-radius: 3px;"></div>
            </div>
            """, unsafe_allow_html=True)

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

    if st.button("Lancer l'EDA Complet", type="primary"):
        with st.spinner("Analyse en cours... Cela peut prendre quelques instants pour les gros datasets..."):
            dataset = st.session_state.get('shared_dataset')

            if dataset is None:
                st.error("Aucun dataset charge. Veuillez d'abord charger un fichier.")
                return

            st.session_state.bus.send_message(Message(
                sender="Frontend",
                receiver="ChefProjet",
                message_type=MessageType.USER_MESSAGE,
                content={'message': 'eda_complet', 'dataset': dataset}
            ))

            response = wait_for_response(max_wait=60)

            if response:
                if response.message_type == MessageType.AGENT_RESPONSE:
                    result = response.content.get('message', '')
                    visualizations = response.content.get('visualizations', {})

                    # Stocker le rapport et les visualisations
                    st.session_state.eda_report = result
                    st.session_state.eda_visualizations = visualizations

                    st.success(" Analyse EDA complete terminee!")
                    st.rerun()

                elif response.message_type == MessageType.ERROR:
                    st.error(response.content.get('error', 'Erreur'))
            else:
                st.error("Delai d'attente depasse. Le dataset est peut-etre trop volumineux.")


def wait_for_response(max_wait=15):
    for _ in range(max_wait * 2):
        time.sleep(0.5)
        response = st.session_state.bus.receive_message("Frontend")
        if response:
            return response
    return None
