"""Page de modélisation"""
import streamlit as st
import time
import pandas as pd
from backend.models.message import Message, MessageType


def render_prediction():
    if not st.session_state.get('dataset_loaded', False):
        st.warning("**Étape précédente requise**")
        st.info("Veuillez d'abord charger un fichier depuis la page d'accueil.")
        if st.button("Retour à l'accueil", type="primary"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return

    workflow = st.session_state.get('workflow', {})
    if not workflow.get('step_2_data_explored', False):
        st.warning("**Étape précédente requise**")
        st.info("Explorez d'abord vos données avant d'entraîner un modèle.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Explorer les données", type="primary", use_container_width=True):
                st.session_state.current_page = "explore"
                st.session_state.current_page_display = "Explorer"
                st.rerun()
        with col2:
            if st.button("Passer cette étape", use_container_width=True):
                st.session_state.workflow['step_2_data_explored'] = True
                st.rerun()
        return

    st.markdown("# Modélisation")
    st.markdown("*Entraînez un modèle ML sur vos données*")
    st.markdown("---")

    if 'prediction_step' not in st.session_state:
        st.session_state.prediction_step = 1
        st.session_state.problem_type = None
        st.session_state.target_column = None

    if st.session_state.prediction_step == 1:
        show_step1_target_selection()
    elif st.session_state.prediction_step == 2:
        show_step2_confirmation_and_detection()
    elif st.session_state.prediction_step == 3:
        show_step3_training()


def show_step1_target_selection():
    st.markdown("## Étape 1/3 : Que voulez-vous prédire ?")
    st.progress(0.33)

    info = st.session_state.dataset_info
    columns = info['column_names']

    default_index = 0
    target_keywords = ['target', 'label', 'class', 'y', 'outcome', 'result', 'price', 'cost', 'amount', 'status', 'churn', 'converted', 'is_', 'has_']

    found_suggestion = False
    for i, col in enumerate(columns):
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in target_keywords):
            if 'id' not in col_lower:
                default_index = i
                found_suggestion = True
                break

    if not found_suggestion and len(columns) > 1:
        default_index = len(columns) - 1

    st.markdown("### Sélectionnez la colonne cible")

    help_text = "C'est la colonne résultat que vous souhaitez estimer."
    if found_suggestion:
        help_text += f"\n*Le système a suggéré : **{columns[default_index]}** *"

    target = st.selectbox("Colonne à prédire :", columns, index=default_index, help=help_text)
    st.session_state.target_column = target

    if st.session_state.get('shared_dataset') is not None:
        col_data = st.session_state.shared_dataset[target]
        st.info(f"**Aperçu :** Cette colonne contient des **{col_data.nunique()}** valeurs différentes.")
        with st.expander("Voir un échantillon des valeurs"):
            st.write(col_data.head(10).to_string(index=False))

    st.markdown("---")

    if st.button("Suivant", type="primary", use_container_width=True):
        st.session_state.prediction_step = 2
        st.rerun()


def show_step2_confirmation_and_detection():
    st.markdown("## Étape 2/3 : Analyse de votre demande")
    st.progress(0.66)

    target = st.session_state.target_column
    dataset = st.session_state.shared_dataset

    detected_type = "unknown"
    explanation = ""

    col_data = dataset[target]

    if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
        detected_type = "classification"
        explanation = "La colonne contient du texte ou des catégories (ex: Oui/Non, Client A/B)."
    elif pd.api.types.is_numeric_dtype(col_data):
        unique_count = col_data.nunique()
        total_count = len(col_data)

        if unique_count < 10 or (unique_count / total_count) < 0.05:
            detected_type = "classification"
            explanation = f"La colonne contient des nombres, mais seulement {unique_count} valeurs différentes."
        else:
            detected_type = "regression"
            explanation = "La colonne contient un grand nombre de valeurs numériques continues."

    st.session_state.problem_type = detected_type

    st.markdown("### Résumé de l'analyse")

    if detected_type == "regression":
        st.success(f"**Type détecté : Prédiction de Valeur (Régression)**")
        st.markdown(f"*Raison :* {explanation}")
        st.markdown("Le système va chercher à prédire un **nombre précis**.")
    elif detected_type == "classification":
        st.success(f"**Type détecté : Prédiction de Catégorie (Classification)**")
        st.markdown(f"*Raison :* {explanation}")
        st.markdown("Le système va chercher à classer les données dans un **groupe**.")
    else:
        st.warning("Impossible de déterminer le type automatiquement.")

    st.info(f"**Colonne cible :** `{target}`")

    with st.expander("Forcer le type (Mode Expert)"):
        manual_type = st.radio(
            "Si la détection est incorrecte, choisissez ici :",
            ["Auto (Recommandé)", "Régression (Nombre)", "Classification (Catégorie)"],
            horizontal=True
        )
        if manual_type != "Auto (Recommandé)":
            if "Régression" in manual_type:
                st.session_state.problem_type = "regression"
            else:
                st.session_state.problem_type = "classification"

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retour"):
            st.session_state.prediction_step = 1
            st.rerun()
    with col2:
        if st.button("Lancer l'entraînement", type="primary", use_container_width=True):
            st.session_state.prediction_step = 3
            st.rerun()


def show_step3_training():
    st.markdown("## Étape 3/3 : Entraînement")
    st.progress(1.0)

    status_placeholder = st.empty()
    result_placeholder = st.empty()

    with status_placeholder:
        with st.spinner("Communication avec l'agent Data Scientist..."):
            time.sleep(0.5)

            dataset = st.session_state.shared_dataset

            msg = Message(
                sender="Frontend",
                receiver="ModelisateurML",
                message_type=MessageType.TASK_REQUEST,
                content={
                    'task': 'entrainer',
                    'dataset': dataset,
                    'target': st.session_state.target_column,
                    'problem_type': st.session_state.problem_type
                }
            )

            st.session_state.bus.send_message(msg)
            status_placeholder.markdown("Recherche du meilleur modèle en cours...")

    response = wait_for_response(max_wait=60)

    if response:
        status_placeholder.empty()

        if response.message_type == MessageType.TASK_RESPONSE:
            with result_placeholder.container():
                st.success("### Analyse Terminée")
                st.markdown(response.content.get('message', ''))

                model_id = response.content.get('model_id')
                if model_id:
                    st.info(f"**Modèle sauvegardé:** `{model_id}`")
                    st.markdown("Vous pouvez maintenant utiliser ce modèle pour faire des prédictions.")

                    if st.button("Aller à la page Prédiction", type="primary", use_container_width=True):
                        st.session_state.current_page = "deploy"
                        st.session_state.current_page_display = "Déployer"
                        st.rerun()

                with st.expander("Données techniques brutes"):
                    st.json(response.content.get('results'))

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Nouvelle Analyse", use_container_width=True):
                        st.session_state.prediction_step = 1
                        st.rerun()
                with col2:
                    if st.button("Explorer les données", use_container_width=True):
                        st.session_state.current_page = "explore"
                        st.session_state.current_page_display = "Explorer"
                        st.rerun()
                with col3:
                    if st.button("Accueil", use_container_width=True):
                        st.session_state.current_page = "home"
                        st.session_state.current_page_display = "Accueil"
                        st.rerun()

        elif response.message_type == MessageType.ERROR:
            status_placeholder.empty()
            with result_placeholder:
                st.error(f"### Une erreur est survenue")
                st.error(response.content.get('error', 'Erreur inconnue'))
                if st.button("Réessayer"):
                    st.session_state.prediction_step = 1
                    st.rerun()
    else:
        status_placeholder.empty()
        st.error("Délai d'attente dépassé. L'agent ne répond pas.")


def wait_for_response(max_wait=30):
    for i in range(max_wait * 10):
        time.sleep(0.1)
        response = st.session_state.bus.receive_message("Frontend")
        if response:
            return response
    return None
