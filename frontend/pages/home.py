"""Page d'accueil"""
import streamlit as st
import io
from backend.models.message import Message, MessageType
from backend.utils.file_detector import FileDetector
from frontend.utils.ui_helpers import page_header, section_title


def render_home():
    page_header("📁", "Chargement des données",
                "Importez votre fichier pour démarrer le pipeline d'analyse",
                badge="Étape 1/5")

    if not st.session_state.dataset_loaded:
        show_upload_section()
    else:
        show_dataset_summary()


def show_upload_section():
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        <div style="background:white;border-radius:16px;padding:28px;
                    border:1px solid #e2e8f0;box-shadow:0 1px 6px rgba(0,0,0,.06);">
            <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">
                Importer un fichier de données
            </div>
            <div style="font-size:13px;color:#64748b;margin-bottom:20px;">
                Glissez-déposez ou cliquez pour sélectionner
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Fichier",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'tsv', 'txt'],
            help=f"Formats supportés : {FileDetector.get_supported_formats_string()}",
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)

    with col2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);
                    border-radius:16px;padding:24px;border:1px solid #bfdbfe;margin-bottom:16px;">
            <div style="font-size:15px;font-weight:700;color:#1d4ed8;margin-bottom:16px;">
                📋 Formats supportés
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:white;border-radius:8px;padding:10px 12px;
                            font-size:13px;color:#1e40af;font-weight:600;
                            border:1px solid #bfdbfe;text-align:center;">CSV / TSV</div>
                <div style="background:white;border-radius:8px;padding:10px 12px;
                            font-size:13px;color:#1e40af;font-weight:600;
                            border:1px solid #bfdbfe;text-align:center;">Excel</div>
                <div style="background:white;border-radius:8px;padding:10px 12px;
                            font-size:13px;color:#1e40af;font-weight:600;
                            border:1px solid #bfdbfe;text-align:center;">JSON</div>
                <div style="background:white;border-radius:8px;padding:10px 12px;
                            font-size:13px;color:#1e40af;font-weight:600;
                            border:1px solid #bfdbfe;text-align:center;">Parquet</div>
            </div>
        </div>
        <div style="background:white;border-radius:16px;padding:20px;border:1px solid #e2e8f0;">
            <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:14px;">
                🤖 Pipeline automatique
            </div>
            <div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #f1f5f9;align-items:center;">
                <span style="font-size:18px;">⚙️</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#1e293b;">Ingénieur Données</div>
                    <div style="font-size:11px;color:#94a3b8;">Charge et valide le fichier</div>
                </div>
            </div>
            <div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #f1f5f9;align-items:center;">
                <span style="font-size:18px;">🔍</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#1e293b;">Analyste</div>
                    <div style="font-size:11px;color:#94a3b8;">Profile et score la qualité</div>
                </div>
            </div>
            <div style="display:flex;gap:12px;padding:8px 0;align-items:center;">
                <span style="font-size:18px;">🎯</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#1e293b;">Chef de Projet</div>
                    <div style="font-size:11px;color:#94a3b8;">Orchestre et coordonne</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _clear_dataset_state():
    """Efface toutes les données liées au dataset précédent."""
    keys = [
        # EDA
        'eda_report', 'eda_visualizations',
        # Preprocessing
        'preprocessing_plan', 'preprocessing_rapport', 'clean_dataset',
        # Modélisation
        'training_result', 'training_in_progress', 'training_start_time',
        'prediction_step', 'problem_type', 'target_column',
        # Chat / RAG
        'chat_history', 'chat_dataset_context', '_rag_indexed_file',
    ]
    for key in keys:
        st.session_state.pop(key, None)
    # Réinitialise le workflow (sauf step_1 qui sera reposé juste après)
    if 'workflow' in st.session_state:
        st.session_state.workflow.update({
            'step_2_data_explored': False,
            'step_3_model_trained': False,
            'step_4_model_deployed': False,
        })


def process_uploaded_file(uploaded_file):
    with st.spinner('Analyse par les agents en cours...'):
        file_data = io.BytesIO(uploaded_file.getvalue())
        filename  = uploaded_file.name

        st.session_state.bus.send_message(Message(
            sender="Frontend", receiver="ChefProjet",
            message_type=MessageType.DATA_UPLOAD,
            content={'file_data': file_data, 'filename': filename}
        ))

        import time
        for _ in range(60):   # 30 secondes max
            time.sleep(0.5)
            response = st.session_state.bus.receive_message("Frontend")
            if response:
                if response.message_type == MessageType.ERROR:
                    st.error(f"❌ {response.content.get('error', 'Erreur inconnue')}")
                    return
                elif response.message_type == MessageType.DATA_VALIDATION:
                    if response.content.get('valid'):
                        # Efface TOUTES les données de l'ancien dataset
                        _clear_dataset_state()
                        st.session_state.dataset_loaded  = True
                        st.session_state.dataset_info    = response.content.get('metadata')
                        st.session_state.shared_dataset  = response.content.get('dataset')
                        st.session_state.dataset_profile = response.content.get('profile')
                        if 'workflow' in st.session_state:
                            st.session_state.workflow['step_1_data_loaded'] = True
                        st.rerun()
                        return
        st.error("⏱️ Délai d'attente dépassé — réessayez.")


def show_dataset_summary():
    info    = st.session_state.dataset_info
    profile = st.session_state.dataset_profile
    quality = info.get('quality_score', 0)
    q_color = "#22c55e" if quality >= 90 else "#f59e0b" if quality >= 70 else "#ef4444"
    q_label = "Excellent" if quality >= 90 else "Bon" if quality >= 70 else "À améliorer"

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);'
        f'border:1px solid #86efac;border-left:4px solid #22c55e;'
        f'border-radius:14px;padding:18px 24px;margin-bottom:24px;'
        f'display:flex;align-items:center;gap:14px;">'
        f'<span style="font-size:28px;">✅</span>'
        f'<div>'
        f'<div style="font-weight:700;color:#15803d;font-size:16px;">Fichier chargé avec succès</div>'
        f'<div style="color:#166534;font-size:13px;margin-top:2px;">'
        f'Les agents ont analysé <strong>{info.get("filename","")}</strong>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Lignes",   f"{info['rows']:,}")
    with c2: st.metric("Colonnes", info['columns'])
    with c3: st.metric("Type",     info['data_type'])
    with c4: st.metric("Format",   info['format'].upper())
    with c5: st.metric("Qualité",  f"{quality}/100", delta=q_label)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown(
            f'<div style="background:white;border-radius:16px;padding:28px;'
            f'border:1px solid #e2e8f0;text-align:center;box-shadow:0 1px 6px rgba(0,0,0,.06);">'
            f'<div style="font-size:11px;color:#64748b;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;">Score de qualité</div>'
            f'<div style="font-size:56px;font-weight:800;color:{q_color};line-height:1;margin-bottom:8px;">{quality}</div>'
            f'<div style="font-size:16px;color:#94a3b8;margin-bottom:16px;">/100</div>'
            f'<div style="background:{q_color}18;color:{q_color};border:1px solid {q_color}40;'
            f'border-radius:20px;padding:6px 14px;font-size:13px;font-weight:700;display:inline-block;">{q_label}</div>'
            f'<div style="margin-top:20px;background:#f1f5f9;border-radius:8px;height:8px;overflow:hidden;">'
            f'<div style="width:{quality}%;height:100%;background:linear-gradient(90deg,{q_color}cc,{q_color});'
            f'border-radius:8px;"></div></div></div>',
            unsafe_allow_html=True
        )

    with col2:
        col_types = profile.get('column_types', {})
        missing   = profile.get('missing_values', {})
        dupes     = profile.get('duplicates', 0)
        stats = [
            ("🔢", "Colonnes numériques",    col_types.get('numeric', 0),    "#2563eb"),
            ("🔤", "Colonnes catégorielles", col_types.get('categorical', 0), "#7c3aed"),
            ("📅", "Colonnes dates",          col_types.get('datetime', 0),   "#0891b2"),
            ("❓", "Valeurs manquantes",      missing.get('total', 0),        "#f59e0b" if missing.get('total', 0) > 0 else "#22c55e"),
            ("📋", "Lignes dupliquées",       dupes,                          "#f59e0b" if dupes > 0 else "#22c55e"),
        ]
        rows_html = ""
        for icon, label, val, color in stats:
            rows_html += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:10px 0;border-bottom:1px solid #f8fafc;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:16px;">{icon}</span>'
                f'<span style="font-size:13px;color:#374151;font-weight:500;">{label}</span>'
                f'</div>'
                f'<span style="background:{color}18;color:{color};border:1px solid {color}30;'
                f'border-radius:8px;padding:3px 10px;font-size:13px;font-weight:700;">{val}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:white;border-radius:16px;padding:24px;'
            f'border:1px solid #e2e8f0;box-shadow:0 1px 6px rgba(0,0,0,.06);">'
            f'<div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:16px;">Profil du dataset</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True
        )

    if profile and profile.get('suggestions'):
        st.markdown("<br>", unsafe_allow_html=True)
        section_title("💡 Suggestions automatiques", "Recommandations générées par les agents")
        cols = st.columns(min(len(profile['suggestions']), 3))
        for i, sugg in enumerate(profile['suggestions'][:3]):
            with cols[i]:
                st.markdown(
                    f'<div style="background:white;border-radius:14px;padding:20px;'
                    f'border:1px solid #e2e8f0;height:100%;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
                    f'<div style="font-size:13px;font-weight:700;color:#1d4ed8;margin-bottom:8px;">{sugg["title"]}</div>'
                    f'<div style="font-size:12px;color:#64748b;line-height:1.6;">{sugg["description"][:200]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        if st.button("🔍 Explorer les données", type="primary", use_container_width=True):
            st.session_state.current_page = "explore"
            st.rerun()
