"""Page d'accueil - Upload de fichier"""
import streamlit as st
import io
from backend.models.message import Message, MessageType
from backend.utils.file_detector import FileDetector


def render_home():
    """Affiche la page d'accueil"""
    
    st.markdown('<h1 class="main-header">Assistant Analyse de Donnees</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analysez vos donnees et creez des modeles predictifs simplement</p>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded:
        show_upload_section()
    else:
        show_dataset_summary()


def show_upload_section():
    """Section d'upload de fichier"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Commencer")
        st.markdown("Pour commencer, uploadez votre fichier de donnees:")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Glissez votre fichier ici",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'tsv', 'txt'],
            help=f"Formats supportes: {FileDetector.get_supported_formats_string()}"
        )
        
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)
    
    with col2:
        st.markdown("### Formats supportes")
        for ext, name in FileDetector.SUPPORTED_FORMATS.items():
            st.markdown(f"- {name} (.{ext})")
        
        st.markdown("---")
        st.markdown("### Aide")
        st.markdown("Premiere fois? Essayez avec un fichier d'exemple pour decouvrir les fonctionnalites.")


def process_uploaded_file(uploaded_file):
    """Traite le fichier uploade"""
    
    with st.spinner('Chargement et analyse du fichier...'):
        # Convertir en bytes pour l'agent
        file_data = io.BytesIO(uploaded_file.getvalue())
        filename = uploaded_file.name
        
        # Envoyer au Chef de Projet
        st.session_state.bus.send_message(Message(
            sender="Frontend",
            receiver="ChefProjet",
            message_type=MessageType.DATA_UPLOAD,
            content={
                'file_data': file_data,
                'filename': filename
            }
        ))
        
        # Attendre la reponse
        import time
        max_attempts = 30
        for _ in range(max_attempts):
            time.sleep(0.5)
            response = st.session_state.bus.receive_message("Frontend")
            
            if response:
                if response.message_type == MessageType.ERROR:
                    st.error(f"Erreur: {response.content.get('error', 'Erreur inconnue')}")
                    return
                
                elif response.message_type == MessageType.DATA_VALIDATION:
                    if response.content.get('valid'):
                        # Stocker les informations ET le dataset
                        st.session_state.dataset_loaded = True
                        st.session_state.dataset_info = response.content.get('metadata')
                        st.session_state.shared_dataset = response.content.get('dataset')
                        st.session_state.dataset_profile = response.content.get('profile')
                        st.session_state.shared_dataset = response.content.get('dataset')  # CRUCIAL!
                        
                        st.success(response.content.get('message', 'Fichier charge avec succes'))
                        
                        # Afficher les suggestions
                        if st.session_state.dataset_profile:
                            suggestions = st.session_state.dataset_profile.get('suggestions', [])
                            if suggestions:
                                st.markdown("### Que voulez-vous faire?")
                                for sugg in suggestions:
                                    with st.expander(f"{sugg['title']}"):
                                        st.markdown(sugg['description'])
                        
                        st.rerun()
                        return
        
        st.error("Delai d'attente depasse")


def show_dataset_summary():
    """Affiche le resume du dataset charge"""
    
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("### Fichier charge avec succes")
    st.markdown('</div>', unsafe_allow_html=True)
    
    info = st.session_state.dataset_info
    profile = st.session_state.dataset_profile
    
    # Metriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lignes", f"{info['rows']:,}")
    with col2:
        st.metric("Colonnes", info['columns'])
    with col3:
        st.metric("Type", info['data_type'])
    with col4:
        st.metric("Format", info['format'].upper())
    
    st.markdown("---")
    
    # Suggestions
    if profile and profile.get('suggestions'):
        st.markdown("### Que voulez-vous faire ensuite?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Explorer les donnees", use_container_width=True):
                st.session_state.current_page = "explore"
                st.session_state.current_page_display = "Explorer"
                st.rerun()
        
        with col2:
            if st.button("Creer un modele predictif", use_container_width=True):
                st.session_state.current_page = "predict"
                st.session_state.current_page_display = "Predire"
                st.rerun()
        
        st.markdown("---")
        
        # Details des suggestions
        st.markdown("### Suggestions automatiques")
        for sugg in profile['suggestions']:
            with st.expander(f"{sugg['title']}"):
                st.markdown(sugg['description'])
