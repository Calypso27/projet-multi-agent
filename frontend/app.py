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
    page_title="DataAgent — Plateforme ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Global — Thème Bleu Professionnel ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', sans-serif !important;
}

.main { background: #f0f4f8; }
.main .block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem;
    max-width: 1300px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2744 100%) !important;
    border-right: 1px solid rgba(30, 64, 175, 0.3);
}
[data-testid="stSidebarContent"] { padding: 0 12px 24px 12px; }
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span { color: #94a3b8 !important; }
[data-testid="stSidebar"] hr {
    border-color: rgba(30, 64, 175, 0.3) !important;
    margin: 10px 0 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
    width: 100%;
    margin: 2px 0;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(37, 99, 235, 0.2) !important;
    border-color: rgba(37, 99, 235, 0.5) !important;
    color: #e2e8f0 !important;
    transform: translateX(2px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border-color: #3b82f6 !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(37, 99, 235, 0.45) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    transform: translateX(2px) !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.55) !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: rgba(255,255,255,0.02) !important;
    border-color: rgba(255,255,255,0.04) !important;
    color: rgba(255,255,255,0.18) !important;
    cursor: not-allowed !important;
}
[data-testid="stSidebar"] [data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1d4ed8, #60a5fa) !important;
    border-radius: 6px !important;
}

/* ── BOUTONS ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    padding: 8px 18px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    color: #1e293b !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #f8fafc !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
}

/* ── MÉTRIQUES ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s;
}
[data-testid="stMetric"]:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.1); }
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 24px !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 500 !important; }

/* ── ONGLETS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #e8edf5;
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-weight: 500;
    font-size: 13px;
    color: #64748b;
    padding: 8px 18px;
    transition: all 0.2s;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1d4ed8 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1) !important;
    font-weight: 600 !important;
}

/* ── ALERTES ── */
.stSuccess {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-left: 4px solid #22c55e !important;
    border-radius: 10px !important;
}
.stWarning {
    background: #fffbeb !important;
    border: 1px solid #fde68a !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 10px !important;
}
.stError {
    background: #fef2f2 !important;
    border: 1px solid #fecaca !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: 10px !important;
}
.stInfo {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-left: 4px solid #2563eb !important;
    border-radius: 10px !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border-color: #e2e8f0 !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}
.stSelectbox > div > div { border-radius: 8px !important; border-color: #e2e8f0 !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #93c5fd !important;
    border-radius: 16px !important;
    background: #eff6ff !important;
    padding: 8px !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563eb !important;
    background: #dbeafe !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #f8fafc !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    border: 1px solid #e2e8f0 !important;
}
.streamlit-expanderContent {
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    background: white !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div {
    border-radius: 6px !important;
    background: #e2e8f0 !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1d4ed8, #60a5fa) !important;
    border-radius: 6px !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-color: #2563eb transparent transparent transparent !important; }

/* ── RADIO ── */
.stRadio > div { gap: 10px; }
.stRadio label {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 14px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 13px !important;
}
.stRadio label:hover { border-color: #2563eb; color: #1d4ed8; }

/* ── DOWNLOAD ── */
.stDownloadButton > button {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    color: #1e293b !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
.stDownloadButton > button:hover {
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
    background: #eff6ff !important;
}
</style>
""", unsafe_allow_html=True)


def initialize_system():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        bus = get_message_bus()
        bus.register_agent("Frontend")

        st.session_state.chef          = ChefProjetAgent()
        st.session_state.data_engineer = DataEngineerAgent()
        st.session_state.analyste      = AnalysteAgent()
        st.session_state.modelisateur  = ModelisateurMLAgent()

        st.session_state.chef.start()
        st.session_state.data_engineer.start()
        st.session_state.analyste.start()
        st.session_state.modelisateur.start()

        st.session_state.bus           = bus
        st.session_state.initialized   = True
        st.session_state.dataset_loaded = False
        st.session_state.current_page  = "home"
        st.session_state.dataset_info  = None

        st.session_state.workflow = {
            'step_1_data_loaded':    False,
            'step_2_data_explored':  False,
            'step_3_model_trained':  False,
            'step_4_model_deployed': False,
        }


def get_workflow_status():
    workflow = st.session_state.get('workflow', {})
    from backend.models.model_manager import ModelManager
    models     = ModelManager.list_models()
    has_models = len(models) > 0
    return {
        'home':    {'unlocked': True,                                        'completed': workflow.get('step_1_data_loaded', False)},
        'explore': {'unlocked': workflow.get('step_1_data_loaded', False),   'completed': workflow.get('step_2_data_explored', False)},
        'predict': {'unlocked': workflow.get('step_2_data_explored', False), 'completed': has_models},
        'deploy':  {'unlocked': has_models,                                  'completed': workflow.get('step_4_model_deployed', False)},
        'summary': {'unlocked': workflow.get('step_2_data_explored', False), 'completed': False},
    }


def main():
    initialize_system()
    workflow_status = get_workflow_status()
    current_page    = st.session_state.get('current_page', 'home')

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:

        # Logo
        st.markdown("""
        <div style="padding: 20px 8px 16px 8px; text-align: center;">
            <div style="font-size: 36px; margin-bottom: 6px;">📊</div>
            <div style="font-size: 17px; font-weight: 800; color: #e2e8f0; letter-spacing: 0.02em;">DataAgent</div>
            <div style="font-size: 11px; color: #475569; font-weight: 500; margin-top: 2px;">Plateforme ML Multi-Agent</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Navigation — Pipeline principal
        pages_config = [
            {"name": "Données",      "key": "home",    "icon": "📁"},
            {"name": "Exploration",  "key": "explore", "icon": "🔍"},
            {"name": "Modélisation", "key": "predict", "icon": "🧠"},
            {"name": "Prédiction",   "key": "deploy",  "icon": "🎯"},
            {"name": "Rapport",      "key": "summary", "icon": "📋"},
        ]

        st.markdown("""
        <div style="font-size: 10px; color: #475569; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.08em;
                    margin: 4px 4px 8px 4px;">Pipeline</div>
        """, unsafe_allow_html=True)

        for i, page_config in enumerate(pages_config):
            status       = workflow_status[page_config['key']]
            is_unlocked  = status['unlocked']
            is_completed = status['completed']
            is_current   = (page_config['key'] == current_page)

            state_badge = " ✓" if is_completed else (" 🔒" if not is_unlocked else "")
            label = f"{page_config['icon']}  {i+1}. {page_config['name']}{state_badge}"

            if is_unlocked:
                btn_type = "primary" if is_current else "secondary"
                if st.button(label, key=f"nav_{page_config['key']}",
                             use_container_width=True, type=btn_type):
                    st.session_state.current_page = page_config['key']
                    st.rerun()
            else:
                st.button(label, key=f"nav_{page_config['key']}",
                          use_container_width=True, disabled=True)

        # Navigation — Chat IA
        st.markdown("""
        <div style="font-size: 10px; color: #475569; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.08em;
                    margin: 12px 4px 8px 4px;">Intelligence Artificielle</div>
        """, unsafe_allow_html=True)

        chat_type = "primary" if current_page == "chat" else "secondary"
        if st.button("💬  Chat avec vos données", key="nav_chat",
                     use_container_width=True, type=chat_type):
            st.session_state.current_page = "chat"
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Progression
        completed_steps = sum(1 for p in pages_config if workflow_status[p['key']]['completed'])
        total_steps     = len(pages_config)

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<span style="font-size:10px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;">Progression</span>'
            f'<span style="font-size:13px;color:#60a5fa;font-weight:700;">{completed_steps}/{total_steps}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(completed_steps / total_steps)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Dataset actuel
        if st.session_state.dataset_loaded and st.session_state.dataset_info:
            info    = st.session_state.dataset_info
            quality = info.get('quality_score', 0)
            q_color = "#22c55e" if quality >= 90 else "#f59e0b" if quality >= 70 else "#ef4444"

            st.markdown(
                f'<div style="background:rgba(255,255,255,0.06);border-radius:10px;padding:12px 14px;border:1px solid rgba(255,255,255,0.08);margin-bottom:8px;">'
                f'<div style="font-size:10px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Dataset actuel</div>'
                f'<div style="font-size:13px;color:#e2e8f0;font-weight:600;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'📄 {info.get("filename", "N/A")}'
                f'</div>'
                f'<div style="display:flex;gap:10px;font-size:12px;color:#94a3b8;margin-bottom:8px;">'
                f'<span>📏 {info.get("rows", 0):,} lignes</span>'
                f'<span>📐 {info.get("columns", 0)} col.</span>'
                f'</div>'
                f'<span style="background:{q_color}20;color:{q_color};border:1px solid {q_color}40;border-radius:12px;padding:2px 8px;font-size:11px;font-weight:600;">'
                f'Qualité {quality}/100'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            if st.button("🔄 Nouveau fichier", use_container_width=True):
                from frontend.pages.home import _clear_dataset_state
                _clear_dataset_state()
                st.session_state.dataset_loaded = False
                st.session_state.dataset_info   = None
                st.session_state.current_page   = "home"
                st.rerun()

            st.markdown("<hr>", unsafe_allow_html=True)

        # ── Backend LLM ───────────────────────────────────────────────────────
        from backend.utils.llm_client import get_backend, list_ollama_models
        st.markdown(
            '<div style="font-size:10px;color:#475569;font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.08em;margin:4px 4px 8px 4px;">Moteur IA</div>',
            unsafe_allow_html=True
        )
        backend = get_backend()

        if backend == "anthropic":
            st.markdown(
                '<div style="background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);'
                'border-radius:8px;padding:8px 12px;font-size:12px;color:#86efac;">'
                '✅ Anthropic API active</div>',
                unsafe_allow_html=True
            )

        elif backend == "ollama":
            st.markdown(
                '<div style="background:rgba(96,165,250,0.12);border:1px solid rgba(96,165,250,0.3);'
                'border-radius:8px;padding:8px 12px;font-size:12px;color:#93c5fd;margin-bottom:8px;">'
                '🦙 Ollama local actif</div>',
                unsafe_allow_html=True
            )
            models = list_ollama_models()
            if models:
                current_model = os.environ.get("OLLAMA_MODEL", models[0])
                selected = st.selectbox(
                    "Modèle",
                    options=models,
                    index=models.index(current_model) if current_model in models else 0,
                    label_visibility="collapsed",
                    key="sidebar_ollama_model"
                )
                if selected != current_model:
                    os.environ["OLLAMA_MODEL"] = selected
                    st.rerun()

        else:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);'
                'border-radius:8px;padding:8px 12px;font-size:12px;color:#fca5a5;margin-bottom:6px;">'
                '⚠️ Aucun LLM disponible</div>',
                unsafe_allow_html=True
            )
            api_key_input = st.text_input(
                "Clé Anthropic",
                type="password",
                placeholder="sk-ant-...",
                label_visibility="collapsed",
                key="sidebar_api_key"
            )
            if api_key_input and api_key_input.startswith("sk-ant-"):
                os.environ["ANTHROPIC_API_KEY"] = api_key_input
                st.rerun()
            st.markdown(
                '<div style="font-size:11px;color:#64748b;margin-top:4px;">'
                'Entrez une clé Anthropic ou lancez Ollama localement.</div>',
                unsafe_allow_html=True
            )
        st.markdown("<hr>", unsafe_allow_html=True)

        # Statut des agents
        from frontend.utils.ui_helpers import agent_status_sidebar
        agent_status_sidebar(agents_running=st.session_state.get('initialized', False))

    # ── PAGES ────────────────────────────────────────────────────────────────
    if current_page == "home":
        from frontend.pages.home import render_home
        render_home()
    elif current_page == "explore":
        from frontend.pages.exploration import render_exploration
        render_exploration()
    elif current_page == "predict":
        from frontend.pages.prediction import render_prediction
        render_prediction()
    elif current_page == "deploy":
        from frontend.pages.deploy import render_deploy
        render_deploy()
    elif current_page == "summary":
        from frontend.pages.summary import render_summary
        render_summary()
    elif current_page == "chat":
        from frontend.pages.chat import render_chat
        render_chat()


if __name__ == "__main__":
    main()
