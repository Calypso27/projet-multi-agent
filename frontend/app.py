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

# Styles CSS
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


def main():
    """Interface principale"""
    initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        
        page = st.radio(
            "Aller à:",
            ["Accueil", "Explorer", "Prédire", "À propos"],
            index=["Accueil", "Explorer", "Prédire", "À propos"].index(
                st.session_state.get('current_page_display', 'Accueil')
            )
        )
        
        page_map = {
            "Accueil": "home",
            "Explorer": "explore",
            "Prédire": "predict",
            "À propos": "about"
        }
        st.session_state.current_page = page_map[page]
        st.session_state.current_page_display = page
        
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
            st.markdown(f"✓ {agent}")
    
    # Afficher la page
    if st.session_state.current_page == "home":
        show_home_page()
    elif st.session_state.current_page == "explore":
        show_explore_page()
    elif st.session_state.current_page == "predict":
        show_predict_page()
    elif st.session_state.current_page == "about":
        show_about_page()


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


def show_about_page():
    """Page à propos"""
    from frontend.pages.about import render_about
    render_about()


if __name__ == "__main__":
    main()
