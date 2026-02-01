"""Application Streamlit principale"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.communication.message_bus import MessageBus, get_message_bus
from backend.agents.chef_projet_agent import ChefProjetAgent
from backend.agents.data_engineer_agent import DataEngineerAgent
from backend.agents.analyste_agent import AnalysteAgent
from backend.agents.modelisateur_ml_agent import ModelisateurMLAgent

st.set_page_config(
    page_title="Plateforme d'Analyse de Données",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none; }
    .workflow-step { padding: 8px 12px; margin: 4px 0; border-radius: 6px; font-size: 14px; }
    .workflow-step.active { background-color: #e3f2fd; border-left: 3px solid #1976d2; font-weight: 600; }
    .workflow-step.completed { color: #2e7d32; }
    .workflow-step.locked { color: #9e9e9e; }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        bus = get_message_bus()
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

        st.session_state.workflow = {
            'step_1_data_loaded': False,
            'step_2_data_explored': False,
            'step_3_model_trained': False,
            'step_4_model_deployed': False,
        }


def get_workflow_status():
    workflow = st.session_state.get('workflow', {})
    from backend.models.model_manager import ModelManager

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
    initialize_system()
    workflow_status = get_workflow_status()

    with st.sidebar:
        st.markdown("### Workflow")
        st.caption("> actuel | + terminé | - verrouillé")
        st.markdown("---")

        pages_config = [
            {"name": "1. Données", "key": "home", "desc": "Charger les données"},
            {"name": "2. Exploration", "key": "explore", "desc": "Analyser la qualité"},
            {"name": "3. Modélisation", "key": "predict", "desc": "Entraîner un modèle"},
            {"name": "4. Prédiction", "key": "deploy", "desc": "Utiliser le modèle"},
            {"name": "5. Rapport", "key": "summary", "desc": "Synthèse finale"},
        ]

        current_page = st.session_state.get('current_page', 'home')

        for page_config in pages_config:
            status = workflow_status[page_config['key']]
            is_unlocked = status['unlocked']
            is_completed = status['completed']
            is_current = (page_config['key'] == current_page)

            if is_current:
                prefix = ">"
            elif is_completed:
                prefix = "+"
            elif not is_unlocked:
                prefix = "-"
            else:
                prefix = " "

            button_label = f"{prefix} {page_config['name']}"

            if is_unlocked:
                button_type = "primary" if is_current else "secondary"
                if st.button(button_label, key=f"nav_{page_config['key']}", use_container_width=True, type=button_type):
                    st.session_state.current_page = page_config['key']
                    st.session_state.current_page_display = page_config['name']
                    st.rerun()
            else:
                st.button(button_label, key=f"nav_{page_config['key']}", use_container_width=True, disabled=True)

        st.markdown("---")

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

    if st.session_state.current_page == "home":
        from frontend.pages.home import render_home
        render_home()
    elif st.session_state.current_page == "explore":
        from frontend.pages.exploration import render_exploration
        render_exploration()
    elif st.session_state.current_page == "predict":
        from frontend.pages.prediction import render_prediction
        render_prediction()
    elif st.session_state.current_page == "deploy":
        from frontend.pages.deploy import render_deploy
        render_deploy()
    elif st.session_state.current_page == "summary":
        from frontend.pages.summary import render_summary
        render_summary()


if __name__ == "__main__":
    main()
