"""Page d'accueil"""
import streamlit as st
import io
from backend.models.message import Message, MessageType
from backend.utils.file_detector import FileDetector


def render_home():
    st.markdown("# Données")
    st.markdown("*Chargez et analysez vos fichiers de données*")

    if not st.session_state.dataset_loaded:
        show_upload_section()
    else:
        show_dataset_summary()


def show_upload_section():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Commencer")
        st.markdown("Pour commencer, uploadez votre fichier de données :")

        uploaded_file = st.file_uploader(
            "Glissez votre fichier ici",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'tsv', 'txt'],
            help=f"Formats supportés : {FileDetector.get_supported_formats_string()}"
        )

        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)

    with col2:
        st.markdown("### Aide")
        st.markdown("Première fois ? Essayez avec un fichier d'exemple pour découvrir les fonctionnalités.")


def process_uploaded_file(uploaded_file):
    with st.spinner('Chargement et analyse du fichier...'):
        file_data = io.BytesIO(uploaded_file.getvalue())
        filename = uploaded_file.name

        st.session_state.bus.send_message(Message(
            sender="Frontend",
            receiver="ChefProjet",
            message_type=MessageType.DATA_UPLOAD,
            content={
                'file_data': file_data,
                'filename': filename
            }
        ))

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
                        st.session_state.dataset_loaded = True
                        st.session_state.dataset_info = response.content.get('metadata')
                        st.session_state.shared_dataset = response.content.get('dataset')
                        st.session_state.dataset_profile = response.content.get('profile')

                        if 'workflow' in st.session_state:
                            st.session_state.workflow['step_1_data_loaded'] = True

                        st.success(response.content.get('message', 'Fichier chargé avec succès'))

                        if st.session_state.dataset_profile:
                            suggestions = st.session_state.dataset_profile.get('suggestions', [])
                            if suggestions:
                                st.markdown("### Que voulez-vous faire ?")
                                for sugg in suggestions:
                                    with st.expander(f"{sugg['title']}"):
                                        st.markdown(sugg['description'])

                        st.rerun()
                        return

        st.error("Délai d'attente dépassé")


def show_dataset_summary():
    st.success("**Fichier chargé avec succès**")

    info = st.session_state.dataset_info
    profile = st.session_state.dataset_profile

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Lignes", f"{info['rows']:,}")
    with col2:
        st.metric("Colonnes", info['columns'])
    with col3:
        st.metric("Type", info['data_type'])
    with col4:
        st.metric("Format", info['format'].upper())
    with col5:
        quality_score = info.get('quality_score', 0)
        if quality_score >= 90:
            quality_delta = "Excellent"
        elif quality_score >= 70:
            quality_delta = "Bon"
        else:
            quality_delta = "Améliorer"
        st.metric("Qualité", f"{quality_score}/100", delta=quality_delta)

    st.markdown("---")

    if profile and profile.get('suggestions'):
        st.markdown("### Que voulez-vous faire ensuite ?")

        if st.button("Explorer les données", use_container_width=True, type="primary"):
            st.session_state.current_page = "explore"
            st.session_state.current_page_display = "Explorer"
            st.rerun()

        st.info("**Conseil**: Explorez d'abord vos données pour comprendre leur structure et leur qualité avant de créer un modèle prédictif.")

        st.markdown("---")

        st.markdown("### Suggestions automatiques")
        for sugg in profile['suggestions']:
            with st.expander(f"{sugg['title']}"):
                st.markdown(sugg['description'])
