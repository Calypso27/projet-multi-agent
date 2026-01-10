import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd

# Ajouter le repertoire parent au path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agents.base_agent import Agent
from agents.agent_manager import AgentManager


# Configuration de la page
st.set_page_config(
    page_title="Multi-Agent System - Monitoring",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalise
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 0.5rem;
    }
    .agent-idle {
        border-left: 4px solid #2ecc71;
    }
    .agent-processing {
        border-left: 4px solid #f39c12;
    }
    .agent-error {
        border-left: 4px solid #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


# Agent de test simple pour demo
class DemoAgent(Agent):
    """Agent de demonstration pour l'interface"""
    
    def __init__(self, name: str, message_bus=None):
        super().__init__(name, message_bus)
        self.tasks_completed = 0
    
    def process_task(self, task: str, payload: dict) -> dict:
        """Traite une tache de demo"""
        self.tasks_completed += 1
        
        if task == "ping":
            return {
                "status": "success",
                "response": "pong",
                "timestamp": datetime.now().isoformat()
            }
        elif task == "compute":
            value = payload.get("value", 0)
            result = value * 2
            return {
                "status": "success",
                "result": result
            }
        else:
            return {
                "status": "error",
                "message": f"Tache inconnue: {task}"
            }


# Initialisation session state
if 'manager' not in st.session_state:
    st.session_state.manager = None
    st.session_state.agents_created = False
    st.session_state.message_log = []


def create_demo_system():
    """Cree un systeme de demo avec 3 agents"""
    manager = AgentManager()
    
    # Creer 3 agents de demo
    agent1 = DemoAgent("Agent_Alpha")
    agent2 = DemoAgent("Agent_Beta")
    agent3 = DemoAgent("Agent_Gamma")
    
    # Enregistrer les agents
    manager.register_agent(agent1)
    manager.register_agent(agent2)
    manager.register_agent(agent3)
    
    # Demarrer les agents
    manager.start_all()
    
    return manager


def send_test_message(manager, from_agent, to_agent, task, payload):
    """Envoie un message de test"""
    agent = manager.get_agent(from_agent)
    if agent:
        agent.send_message(to_agent, task, payload)
        
        # Logger le message
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "from": from_agent,
            "to": to_agent,
            "task": task,
            "payload": str(payload)
        }
        st.session_state.message_log.append(log_entry)


# HEADER
st.markdown('<h1 class="main-header">Systeme Multi-Agent - Monitoring</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# SIDEBAR - Controles
with st.sidebar:
    st.header("Controles Systeme")
    
    # Creer le systeme
    if st.button("Initialiser Systeme", type="primary", use_container_width=True):
        with st.spinner("Initialisation du systeme..."):
            st.session_state.manager = create_demo_system()
            st.session_state.agents_created = True
            st.success("Systeme initialise avec succes")
    
    # Reset
    if st.button("Reinitialiser", use_container_width=True):
        if st.session_state.manager:
            st.session_state.manager.shutdown()
        st.session_state.manager = None
        st.session_state.agents_created = False
        st.session_state.message_log = []
        st.rerun()
    
    st.markdown("---")
    
    # Actions rapides
    st.subheader("Actions Rapides")
    
    if st.session_state.agents_created and st.session_state.manager:
        if st.button("Test Ping", use_container_width=True):
            send_test_message(
                st.session_state.manager,
                "Agent_Alpha",
                "Agent_Beta",
                "ping",
                {}
            )
            st.rerun()
        
        if st.button("Test Compute", use_container_width=True):
            send_test_message(
                st.session_state.manager,
                "Agent_Beta",
                "Agent_Gamma",
                "compute",
                {"value": 42}
            )
            st.rerun()
    
    st.markdown("---")
    
    # Info projet
    st.subheader("Info Projet")
    st.text("Semaine : 1/6")
    st.text("Phase : Architecture")
    st.text("Date : 06/01/2026")


# MAIN CONTENT
if not st.session_state.agents_created:
    # Message d'accueil
    st.info("Cliquez sur 'Initialiser Systeme' pour demarrer la demonstration")
    
    st.subheader("A propos")
    st.write("""
    Cette interface de monitoring permet de visualiser le fonctionnement du socle multi-agent.
    
    **Fonctionnalites:**
    - Visualisation des agents en temps reel
    - Suivi des messages echanges
    - Statistiques du systeme
    - Tests interactifs
    
    **Agents de demonstration:**
    - Agent_Alpha
    - Agent_Beta
    - Agent_Gamma
    """)

else:
    manager = st.session_state.manager
    
    # METRIQUES
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Agents Actifs", len(manager.agents))
    
    with col2:
        bus_stats = manager.message_bus.get_stats()
        st.metric("Messages Totaux", bus_stats['total_messages_sent'])
    
    with col3:
        pending = sum(bus_stats['pending_messages'].values())
        st.metric("Messages en Attente", pending)
    
    with col4:
        uptime = (datetime.now() - manager.start_time).seconds
        st.metric("Uptime (s)", uptime)
    
    st.markdown("---")
    
    # ONGLETS
    tab1, tab2, tab3, tab4 = st.tabs([
        "Agents",
        "Messages", 
        "Statistiques",
        "Envoi Manuel"
    ])
    
    # TAB 1: Agents
    with tab1:
        st.subheader("Etat des Agents")
        
        status = manager.get_all_status()
        
        for agent_name, agent_status in status.items():
            state = agent_status.get('state', 'unknown')
            
            # Determiner la classe CSS
            if state == 'idle':
                css_class = 'agent-idle'
                icon = "🟢"
            elif state == 'processing':
                css_class = 'agent-processing'
                icon = "🟡"
            else:
                css_class = 'agent-error'
                icon = "🔴"
            
            # Afficher la carte d'agent
            st.markdown(f"""
            <div class="agent-card {css_class}">
                <strong>{icon} {agent_name}</strong><br>
                Etat: {state}<br>
                Timestamp: {agent_status.get('timestamp', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 2: Messages
    with tab2:
        st.subheader("Flux de Messages")
        
        if st.session_state.message_log:
            # Afficher sous forme de tableau
            df = pd.DataFrame(st.session_state.message_log)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Graphique des messages par agent
            st.subheader("Messages par Agent")
            
            from_counts = df['from'].value_counts()
            st.bar_chart(from_counts)
        else:
            st.info("Aucun message envoye pour le moment")
    
    # TAB 3: Statistiques
    with tab3:
        st.subheader("Statistiques du Systeme")
        
        system_info = manager.get_system_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informations Generales**")
            st.write(f"Demarrage: {system_info['start_time']}")
            st.write(f"Uptime: {system_info['uptime_seconds']:.2f}s")
            st.write(f"Agents: {system_info['registered_agents']}")
        
        with col2:
            st.markdown("**MessageBus**")
            bus_stats = system_info['message_bus_stats']
            st.write(f"Messages totaux: {bus_stats['total_messages_sent']}")
            st.write(f"Agents enregistres: {bus_stats['registered_agents']}")
        
        st.markdown("**Messages en Attente par Agent**")
        pending_df = pd.DataFrame(
            list(bus_stats['pending_messages'].items()),
            columns=['Agent', 'Messages']
        )
        st.dataframe(pending_df, hide_index=True)
        
        # Historique complet
        st.markdown("**Historique Complet**")
        history = manager.message_bus.get_history()
        
        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun message dans l'historique")
    
    # TAB 4: Envoi Manuel
    with tab4:
        st.subheader("Envoyer un Message Manuel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            agent_names = list(manager.agents.keys())
            from_agent = st.selectbox("De (Agent)", agent_names, key="from")
            to_agent = st.selectbox("Vers (Agent)", agent_names, key="to")
        
        with col2:
            task = st.selectbox("Tache", ["ping", "compute", "custom"])
            
            if task == "compute":
                value = st.number_input("Valeur", value=10)
                payload = {"value": value}
            elif task == "custom":
                payload_str = st.text_input("Payload (JSON)", '{"key": "value"}')
                try:
                    import json
                    payload = json.loads(payload_str)
                except:
                    payload = {}
                    st.error("JSON invalide")
            else:
                payload = {}
        
        if st.button("Envoyer Message", type="primary"):
            if from_agent and to_agent:
                send_test_message(
                    manager,
                    from_agent,
                    to_agent,
                    task,
                    payload
                )
                st.success(f"Message envoye: {from_agent} -> {to_agent}")
                st.rerun()
            else:
                st.error("Selectionnez les agents")


# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Plateforme Conversationnelle d'Analyse de Donnees - Semaine 1</small><br>
    <small>Systeme Multi-Agent - Interface de Monitoring</small>
</div>
""", unsafe_allow_html=True)