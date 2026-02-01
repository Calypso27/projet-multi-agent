"""Application Streamlit - Interface principale avec navigation multi-pages"""
import streamlit as st
import sys
import os

# Ajouter le chemin backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.communication.message_bus import MessageBus, get_message_bus
from backend.agents.chef_projet_agent import ChefProjetAgent
from backend.agents.data_engineer_agent import DataEngineerAgent
from backend.agents.analyste_agent import AnalysteAgent
from backend.agents.modelisateur_ml_agent import ModelisateurMLAgent

# Configuration de la page
st.set_page_config(
    page_title="Plateforme d'Analyse de Données",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS + Masquer le menu de navigation par défaut de Streamlit
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 4px;
        margin: 1rem 0;
    }
    /* Masquer le menu de navigation par défaut */
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    """Initialise le système multi-agent"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        bus = get_message_bus()

        # Enregistrer Frontend comme agent
        bus.register_agent("Frontend")

        st.session_state.chef = ChefProjetAgent()
        st.session_state.data_engineer = DataEngineerAgent()
        st.session_state.analyste = AnalysteAgent()
        st.session_state.modelisateur = ModelisateurMLAgent()

        st.session_state.chef.start()
        st.session_state.data_engineer.start()
        st.session_state.analyste.start()
        st.session_state.modelisateur.start()

        st.session_state.bus = bus
        st.session_state.initialized = True
        st.session_state.dataset_loaded = False
        st.session_state.current_page = "home"
        st.session_state.dataset_info = None

        # Workflow séquentiel - état des étapes
        st.session_state.workflow = {
            'step_1_data_loaded': False,      # Étape 1: Données chargées
            'step_2_data_explored': False,    # Étape 2: Données explorées
            'step_3_model_trained': False,    # Étape 3: Modèle entraîné
            'step_4_model_deployed': False,   # Étape 4: Modèle déployé (prédictions faites)
        }


def get_workflow_status():
    """Retourne le statut du workflow pour chaque page"""
    workflow = st.session_state.get('workflow', {})
    from backend.models.model_manager import ModelManager

    # Vérifier si des modèles existent
    models = ModelManager.list_models()
    has_models = len(models) > 0

    return {
        'home': {'unlocked': True, 'completed': workflow.get('step_1_data_loaded', False)},
        'explore': {'unlocked': workflow.get('step_1_data_loaded', False), 'completed': workflow.get('step_2_data_explored', False)},
        'predict': {'unlocked': workflow.get('step_2_data_explored', False), 'completed': has_models},
        'deploy': {'unlocked': has_models, 'completed': workflow.get('step_4_model_deployed', False)},
        'summary': {'unlocked': workflow.get('step_2_data_explored', False), 'completed': False}
    }


def main():
    """Interface principale"""
    initialize_system()

    # Obtenir le statut du workflow
    workflow_status = get_workflow_status()

    # Sidebar
    with st.sidebar:
        st.markdown("### Workflow")
        st.markdown("---")

        # Définition des pages avec leur ordre dans le workflow
        pages_config = [
            {"name": "1. Données", "key": "home", "desc": "Charger les données"},
            {"name": "2. Exploration", "key": "explore", "desc": "Analyser la qualité"},
            {"name": "3. Modélisation", "key": "predict", "desc": "Entraîner un modèle"},
            {"name": "4. Prédiction", "key": "deploy", "desc": "Utiliser le modèle"},
            {"name": "5. Rapport", "key": "summary", "desc": "Synthèse finale"},
        ]

        # Afficher chaque étape du workflow
        for page_config in pages_config:
            status = workflow_status[page_config['key']]
            is_unlocked = status['unlocked']
            is_completed = status['completed']

            # Déterminer l'icône de statut
            if is_completed:
                status_icon = "[OK]"
            elif is_unlocked:
                status_icon = ""
            else:
                status_icon = "[verrouillé]"

            # Style du bouton selon l'état
            if is_unlocked:
                # Page accessible
                button_label = f"{page_config['name']} {status_icon}".strip()
                if st.button(button_label, key=f"nav_{page_config['key']}", use_container_width=True):
                    st.session_state.current_page = page_config['key']
                    st.session_state.current_page_display = page_config['name']
                    st.rerun()

            else:
                # Page verrouillée
                st.button(
                    f"{page_config['name']} {status_icon}",
                    key=f"nav_{page_config['key']}",
                    use_container_width=True,
                    disabled=True
                )

        st.markdown("---")

        # Indicateur de progression
        completed_steps = sum(1 for p in pages_config if workflow_status[p['key']]['completed'])
        total_steps = len(pages_config)
        progress = completed_steps / total_steps

        st.markdown("### Progression")
        st.progress(progress)
        st.markdown(f"**{completed_steps}/{total_steps}** étapes")

        st.markdown("---")
        
        if st.session_state.dataset_loaded and st.session_state.dataset_info:
            st.markdown("### Dataset Actuel")
            info = st.session_state.dataset_info
            st.markdown(f"**Fichier:** {info.get('filename', 'N/A')}")
            st.markdown(f"**Lignes:** {info.get('rows', 0):,}")
            st.markdown(f"**Colonnes:** {info.get('columns', 0)}")
            
            if st.button("Nouveau fichier"):
                st.session_state.dataset_loaded = False
                st.session_state.dataset_info = None
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Système")
        agents = ["Chef de Projet", "Ingénieur Données", "Analyste", "Modélisateur ML"]
        for agent in agents:
            st.markdown(f"- {agent}")
    
    # Afficher la page
    if st.session_state.current_page == "home":
        show_home_page()
    elif st.session_state.current_page == "explore":
        show_explore_page()
    elif st.session_state.current_page == "predict":
        show_predict_page()
    elif st.session_state.current_page == "deploy":
        show_deploy_page()
    elif st.session_state.current_page == "summary":
        show_summary_page()


def show_home_page():
    """Page d'accueil"""
    from frontend.pages.home import render_home
    render_home()


def show_explore_page():
    """Page d'exploration"""
    from frontend.pages.exploration import render_exploration
    render_exploration()


def show_predict_page():
    """Page de prédiction"""
    from frontend.pages.prediction import render_prediction
    render_prediction()


def show_deploy_page():
    """Page de déploiement"""
    from frontend.pages.deploy import render_deploy
    render_deploy()


def show_summary_page():
    """Page executive summary"""
    from frontend.pages.summary import render_summary
    render_summary()


if __name__ == "__main__":
    main()
