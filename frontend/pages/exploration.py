"""Page d'exploration - Analyse des donnees"""
import streamlit as st
import time
from backend.models.message import Message, MessageType


def render_exploration():
    """Page d'exploration des donnees"""
    
    if not st.session_state.dataset_loaded:
        st.warning("Veuillez d'abord charger un fichier depuis la page d'accueil")
        if st.button("Retour a l'accueil"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return
    
    st.markdown("# Exploration des Donnees")
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["Vue Generale", "Statistiques", "Correlations"])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_statistics_tab()
    
    with tab3:
        show_correlations_tab()


def show_overview_tab():
    """Onglet vue generale"""
    
    st.markdown("### Informations du Dataset")
    
    info = st.session_state.dataset_info
    profile = st.session_state.dataset_profile
    
    # Metriques principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", f"{info['rows']:,}")
    with col2:
        st.metric("Colonnes", info['columns'])
    with col3:
        missing = profile['missing_values']['total']
        st.metric("Valeurs manquantes", f"{missing:,}")
    
    # Types de colonnes
    st.markdown("### Repartition des Types de Colonnes")
    col_types = profile['column_types']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numeriques", col_types['numeric'])
    with col2:
        st.metric("Categorielles", col_types['categorical'])
    with col3:
        st.metric("Dates", col_types['datetime'])
    
    # Details des colonnes
    st.markdown("### Details des Colonnes")
    
    column_data = []
    for col in info['column_names']:
        dtype = info['dtypes'][col]
        column_data.append({
            'Nom': col,
            'Type': dtype
        })
    
    st.dataframe(column_data, use_container_width=True)
    
    # Qualite des donnees
    if profile['missing_values']['total'] > 0 or profile['duplicates'] > 0:
        st.markdown("### Qualite des Donnees")
        
        if profile['missing_values']['total'] > 0:
            st.warning(f"Attention: {profile['missing_values']['total']} valeurs manquantes detectees ({profile['missing_values']['percentage']:.1f}%)")
        
        if profile['duplicates'] > 0:
            st.warning(f"Attention: {profile['duplicates']} lignes dupliquees")


def show_statistics_tab():
    """Onglet statistiques"""
    
    st.markdown("### Statistiques Descriptives")
    
    if st.button("Calculer les statistiques detaillees"):
        with st.spinner("Analyse en cours..."):
            # Récupérer le dataset depuis session_state
            dataset = st.session_state.get('shared_dataset')
            
            # Demander au Chef de Projet
            st.session_state.bus.send_message(Message(
                sender="Frontend",
                receiver="ChefProjet",
                message_type=MessageType.USER_MESSAGE,
                content={'message': 'statistiques', 'dataset': dataset}
            ))
            
            # Attendre la reponse
            response = wait_for_response()
            
            if response:
                if response.message_type == MessageType.AGENT_RESPONSE:
                    st.markdown(response.content.get('message', ''))
                elif response.message_type == MessageType.ERROR:
                    st.error(response.content.get('error', 'Erreur'))
            else:
                st.error("Delai d'attente depasse")


def show_correlations_tab():
    """Onglet correlations"""
    
    st.markdown("### Analyse Complete")
    
    if st.button("Lancer l'analyse complete"):
        with st.spinner("Analyse en cours..."):
            # Récupérer le dataset
            dataset = st.session_state.get('shared_dataset')
            
            # Demander l'analyse complete
            st.session_state.bus.send_message(Message(
                sender="Frontend",
                receiver="ChefProjet",
                message_type=MessageType.USER_MESSAGE,
                content={'message': 'analyser', 'dataset': dataset}
            ))
            
            response = wait_for_response(max_wait=30)
            
            if response:
                if response.message_type == MessageType.AGENT_RESPONSE:
                    st.markdown(response.content.get('message', ''))
                elif response.message_type == MessageType.ERROR:
                    st.error(response.content.get('error', 'Erreur'))
            else:
                st.error("Delai d'attente depasse")


def wait_for_response(max_wait=15):
    """Attend une reponse de l'agent"""
    for _ in range(max_wait * 2):
        time.sleep(0.5)
        response = st.session_state.bus.receive_message("Frontend")
        if response:
            return response
    return None
