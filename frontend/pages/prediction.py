"""Page de prediction - Assistant guide (Version Automatique / Layman Friendly)"""
import streamlit as st
import time
import pandas as pd
from backend.models.message import Message, MessageType


def render_prediction():
    """Page de prediction guidée - Version Automatique"""

    # Vérifier si les données sont chargées
    if not st.session_state.get('dataset_loaded', False):
        st.warning("**Étape précédente requise**")
        st.info("Veuillez d'abord charger un fichier depuis la page d'accueil.")
        if st.button("Retour à l'accueil", type="primary"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return

    # Vérifier si l'exploration a été faite
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
    
    # Initialiser l'état
    if 'prediction_step' not in st.session_state:
        st.session_state.prediction_step = 1
        st.session_state.problem_type = None # Sera déterminé auto
        st.session_state.target_column = None
    
    # Nouveau flux simplifié
    if st.session_state.prediction_step == 1:
        show_step1_target_selection()
    elif st.session_state.prediction_step == 2:
        show_step2_confirmation_and_detection()
    elif st.session_state.prediction_step == 3:
        show_step3_training()


def show_step1_target_selection():
    """
    Etape 1 : L'utilisateur choisit simplement ce qu'il veut prédire.
    Amélioration : Détection automatique de la colonne la plus probable.
    """
    
    st.markdown("## Étape 1/3 : Que voulez-vous prédire ?")
    st.progress(0.33)
    
    info = st.session_state.dataset_info
    columns = info['column_names']
    
    # --- LOGIQUE DE SUGGESTION AUTOMATIQUE ---
    default_index = 0
    
    # Liste de mots-clés qui indiquent souvent une colonne cible
    target_keywords = ['target', 'label', 'class', 'y', 'outcome', 'result', 'price', 'cost', 'amount', 'status', 'churn', 'converted', 'is_', 'has_']
    
    found_suggestion = False
    for i, col in enumerate(columns):
        col_lower = col.lower()
        # On cherche un mot clé dans le nom de la colonne
        if any(keyword in col_lower for keyword in target_keywords):
            # On évite les colonnes ID (ex: customer_id contient 'id')
            if 'id' not in col_lower: 
                default_index = i
                found_suggestion = True
                break # On prend la première correspondance
    
    # Si rien n'a été trouvé, on suggère souvent la dernière colonne (convention courante dans les datasets académiques)
    if not found_suggestion and len(columns) > 1:
        default_index = len(columns) - 1

    # --- AFFICHAGE ---
    
    st.markdown("### Sélectionnez la colonne cible")
    
    help_text = "C'est la colonne résultat que vous souhaitez estimer."
    if found_suggestion:
        help_text += f"\n*Le système a suggéré : **{columns[default_index]}** *"
    
    target = st.selectbox(
        "Colonne à prédire :",
        columns,
        index=default_index, # On applique la suggestion calculée
        help=help_text
    )
    
    st.session_state.target_column = target
    
    # Petit aperçu des données de cette colonne pour aider l'utilisateur
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
    """
    Etape 2 : Le système détecte automatiquement le type de problème
    et demande confirmation.
    """
    
    st.markdown("## Étape 2/3 : Analyse de votre demande")
    st.progress(0.66)
    
    target = st.session_state.target_column
    dataset = st.session_state.shared_dataset
    
    # --- LOGIQUE DE DÉTECTION AUTOMATIQUE ---
    detected_type = "unknown"
    explanation = ""
    
    col_data = dataset[target]
    
    # Cas 1 : C'est du texte (Object) -> Classification
    if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
        detected_type = "classification"
        explanation = "La colonne contient du texte ou des catégories (ex: Oui/Non, Client A/B)."
    
    # Cas 2 : C'est numérique
    elif pd.api.types.is_numeric_dtype(col_data):
        unique_count = col_data.nunique()
        total_count = len(col_data)
        
        # Si peu de valeurs uniques (par ex. < 10 ou < 5% du total) -> Classification
        if unique_count < 10 or (unique_count / total_count) < 0.05:
            detected_type = "classification"
            explanation = f"La colonne contient des nombres, mais seulement {unique_count} valeurs différentes (ex: notes 1 à 5)."
        else:
            # Sinon -> Régression
            detected_type = "regression"
            explanation = "La colonne contient un grand nombre de valeurs numériques continues (ex: Prix, Salaire, Température)."
    
    # Stocker le résultat détecté
    st.session_state.problem_type = detected_type
    
    # --- AFFICHAGE UTILISATEUR ---
    
    st.markdown("### Résumé de l'analyse")
    
    # On utilise des boîtes visuelles pour expliquer simplement
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

    # Option pour l'utilisateur expert : permettre de changer si la détection est mauvaise
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
    """Etape 3: Entrainement"""
    
    st.markdown("## Étape 3/3 : Entraînement")
    st.progress(1.0)
    
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    
    with status_placeholder:
        with st.spinner("Communication avec l'agent Data Scientist..."):
            time.sleep(0.5)
            
            dataset = st.session_state.shared_dataset
            
            # Envoi du message avec le type détecté automatiquement
            msg = Message(
                sender="Frontend",
                receiver="ModelisateurML",
                message_type=MessageType.TASK_REQUEST,
                content={
                    'task': 'entrainer',
                    'dataset': dataset,
                    'target': st.session_state.target_column,
                    'problem_type': st.session_state.problem_type # On utilise le type auto-détecté
                }
            )

            print(f"[DEBUG] Envoi du message vers ModelisateurML: task=entrainer, target={st.session_state.target_column}, problem_type={st.session_state.problem_type}")
            print(f"[DEBUG] Dataset shape: {dataset.shape if hasattr(dataset, 'shape') else 'N/A'}")

            success = st.session_state.bus.send_message(msg)
            print(f"[DEBUG] Message envoyé avec succès: {success}")

            status_placeholder.markdown("⏳ Recherche du meilleur modèle en cours...")

    response = wait_for_response(max_wait=60)

    if response:
        status_placeholder.empty()
        
        if response.message_type == MessageType.TASK_RESPONSE:
            with result_placeholder.container():
                st.success("### Analyse Terminée")
                
                # Affichage du résultat
                st.markdown(response.content.get('message', ''))

                # Si un modèle a été sauvegardé
                model_id = response.content.get('model_id')
                if model_id:
                    st.info(f"**Modèle sauvegardé:** `{model_id}`")
                    st.markdown("Vous pouvez maintenant utiliser ce modèle pour faire des prédictions sur de nouvelles données.")

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
    """Attend une réponse avec debug amélioré"""
    for i in range(max_wait * 10):
        time.sleep(0.1)
        response = st.session_state.bus.receive_message("Frontend")
        if response:
            print(f"[DEBUG] Réponse reçue: {response}")
            return response

        # Log tous les 5 secondes
        if i % 50 == 0 and i > 0:
            print(f"[DEBUG] Attente... {i/10:.0f}s écoulées")

    print(f"[DEBUG] Timeout après {max_wait}s - Aucune réponse reçue")
    return None