"""Page de prediction - Assistant guide"""
import streamlit as st
import time
from backend.models.message import Message, MessageType


def render_prediction():
    """Page de prediction guidee"""
    
    if not st.session_state.dataset_loaded:
        st.warning("Veuillez d'abord charger un fichier depuis la page d'accueil")
        if st.button("Retour a l'accueil"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return
    
    st.markdown("# Assistant de Prediction")
    
    # Initialiser l'etat du wizard
    if 'prediction_step' not in st.session_state:
        st.session_state.prediction_step = 1
        st.session_state.problem_type = None
        st.session_state.target_column = None
    
    # Afficher l'etape actuelle
    if st.session_state.prediction_step == 1:
        show_step1_problem_type()
    elif st.session_state.prediction_step == 2:
        show_step2_target_selection()
    elif st.session_state.prediction_step == 3:
        show_step3_confirmation()
    elif st.session_state.prediction_step == 4:
        show_step4_training()


def show_step1_problem_type():
    """Etape 1: Choix du type de probleme"""
    
    st.markdown("## Etape 1/4: Type de Prediction")
    st.progress(0.25)
    
    st.markdown("### Que voulez-vous predire?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Un nombre", use_container_width=True, help="Ex: prix, temperature, quantite"):
            st.session_state.problem_type = "regression"
            st.session_state.prediction_step = 2
            st.rerun()
        
        st.markdown("""
        **Exemples:**
        - Prix futurs
        - Temperature
        - Ventes
        - Note/Score
        """)
    
    with col2:
        if st.button("Une categorie", use_container_width=True, help="Ex: oui/non, type, classe"):
            st.session_state.problem_type = "classification"
            st.session_state.prediction_step = 2
            st.rerun()
        
        st.markdown("""
        **Exemples:**
        - Oui/Non
        - Succes/Echec
        - Type de client
        - Categorie de produit
        """)


def show_step2_target_selection():
    """Etape 2: Selection de la variable cible"""
    
    st.markdown("## Etape 2/4: Variable a Predire")
    st.progress(0.50)
    
    problem_label = "nombre" if st.session_state.problem_type == "regression" else "categorie"
    st.markdown(f"### Quelle colonne voulez-vous predire? ({problem_label})")
    
    info = st.session_state.dataset_info
    profile = st.session_state.dataset_profile
    
    # Filtrer les colonnes appropriees
    if st.session_state.problem_type == "regression":
        available_columns = profile['numeric_columns']
        help_text = "Selectionnez une colonne numerique"
    else:
        available_columns = info['column_names']
        help_text = "Selectionnez la colonne a classer"
    
    if not available_columns:
        st.error(f"Aucune colonne appropriee pour ce type de prediction")
        if st.button("Retour"):
            st.session_state.prediction_step = 1
            st.rerun()
        return
    
    target = st.selectbox(
        "Colonne cible:",
        available_columns,
        help=help_text
    )
    
    st.session_state.target_column = target
    
    # Info sur la colonne selectionnee
    st.markdown(f"**Type:** {info['dtypes'][target]}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retour"):
            st.session_state.prediction_step = 1
            st.rerun()
    with col2:
        if st.button("Suivant"):
            st.session_state.prediction_step = 3
            st.rerun()


def show_step3_confirmation():
    """Etape 3: Confirmation"""
    
    st.markdown("## Etape 3/4: Confirmation")
    st.progress(0.75)
    
    st.markdown("### Recapitulatif")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    
    problem_label = "Regression (prediction de nombre)" if st.session_state.problem_type == "regression" else "Classification (prediction de categorie)"
    
    st.markdown(f"""
    - **Type:** {problem_label}
    - **Variable cible:** {st.session_state.target_column}
    - **Dataset:** {st.session_state.dataset_info['rows']} lignes
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Configuration")
    st.markdown("""
    Le systeme va automatiquement:
    1. Preparer les donnees
    2. Diviser en ensemble d'entrainement (80%) et test (20%)
    3. Tester plusieurs modeles de machine learning
    4. Selectionner le meilleur modele
    5. Evaluer les performances
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retour"):
            st.session_state.prediction_step = 2
            st.rerun()
    with col2:
        if st.button("Lancer l'entrainement"):
            st.session_state.prediction_step = 4
            st.rerun()


def show_step4_training():
    """Etape 4: Entrainement"""
    
    st.markdown("## Etape 4/4: Entrainement")
    st.progress(1.0)
    
    with st.spinner("Entrainement des modeles en cours..."):
        # Récupérer le dataset
        dataset = st.session_state.get('shared_dataset')
        
        # Envoyer la demande d'entrainement
        st.session_state.bus.send_message(Message(
            sender="Frontend",
            receiver="ChefProjet",
            message_type=MessageType.USER_MESSAGE,
            content={
                'message': f'entrainer {st.session_state.target_column}',
                'target': st.session_state.target_column,
                'problem_type': st.session_state.problem_type,
                'dataset': dataset
            }
        ))
        
        # Attendre la reponse
        response = wait_for_response(max_wait=60)
        
        if response:
            if response.message_type == MessageType.AGENT_RESPONSE:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### Entrainement termine avec succes")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(response.content.get('message', ''))
                
                # Boutons d'action
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Nouvelle prediction"):
                        st.session_state.prediction_step = 1
                        st.session_state.problem_type = None
                        st.session_state.target_column = None
                        st.rerun()
                
                with col2:
                    if st.button("Retour a l'accueil"):
                        st.session_state.current_page = "home"
                        st.session_state.current_page_display = "Accueil"
                        st.session_state.prediction_step = 1
                        st.rerun()
            
            elif response.message_type == MessageType.ERROR:
                st.error(f"Erreur: {response.content.get('error', 'Erreur inconnue')}")
                
                if st.button("Reessayer"):
                    st.session_state.prediction_step = 1
                    st.rerun()
        else:
            st.error("Delai d'attente depasse")
            
            if st.button("Reessayer"):
                st.session_state.prediction_step = 1
                st.rerun()


def wait_for_response(max_wait=30):
    """Attend une reponse de l'agent"""
    for _ in range(max_wait * 2):
        time.sleep(0.5)
        response = st.session_state.bus.receive_message("Frontend")
        if response:
            return response
    return None
