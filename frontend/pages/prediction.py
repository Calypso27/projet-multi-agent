"""Page de modélisation"""
import streamlit as st
import time
import pandas as pd
from backend.models.message import Message, MessageType
from frontend.utils.ui_helpers import page_header, step_indicator


def render_prediction():
    if not st.session_state.get('dataset_loaded', False):
        st.warning("**Étape précédente requise**")
        st.info("Veuillez d'abord charger un fichier depuis la page d'accueil.")
        if st.button("Retour à l'accueil", type="primary"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return

    workflow = st.session_state.get('workflow', {})
    if not workflow.get('step_2_data_explored', False):
        st.warning("**Étape précédente requise**")
        st.info("Explorez d'abord vos données avant d'entraîner un modèle.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Explorer les données", type="primary", use_container_width=True):
                st.session_state.current_page = "explore"
                st.session_state.current_page_display = "Explorer"
                st.rerun()
        with col2:
            if st.button("Passer cette étape", use_container_width=True):
                st.session_state.workflow['step_2_data_explored'] = True
                st.rerun()
        return

    page_header("🧠", "Modélisation ML",
                "Entraînez et comparez plusieurs algorithmes automatiquement",
                badge="Étape 3/5")

    if 'prediction_step' not in st.session_state:
        st.session_state.prediction_step = 1
        st.session_state.problem_type = None
        st.session_state.target_column = None

    step_indicator(
        current=st.session_state.prediction_step,
        total=3,
        labels=["Variable cible", "Analyse", "Entraînement"]
    )

    if st.session_state.prediction_step == 1:
        show_step1_target_selection()
    elif st.session_state.prediction_step == 2:
        show_step2_confirmation_and_detection()
    elif st.session_state.prediction_step == 3:
        show_step3_training()


def show_step1_target_selection():
    st.markdown(
        '<div style="background:white;border-radius:14px;padding:24px;'
        'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);'
        'margin-bottom:20px;">'
        '<div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">'
        'Que souhaitez-vous prédire ?'
        '</div>'
        '<div style="font-size:13px;color:#64748b;">'
        'Sélectionnez la colonne résultat que le modèle devra estimer'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    info = st.session_state.dataset_info
    columns = info['column_names']

    default_index = 0
    target_keywords = ['target', 'label', 'class', 'y', 'outcome', 'result', 'price', 'cost', 'amount', 'status', 'churn', 'converted', 'is_', 'has_']

    found_suggestion = False
    for i, col in enumerate(columns):
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in target_keywords):
            if 'id' not in col_lower:
                default_index = i
                found_suggestion = True
                break

    if not found_suggestion and len(columns) > 1:
        default_index = len(columns) - 1

    st.markdown("### Sélectionnez la colonne cible")

    help_text = "C'est la colonne résultat que vous souhaitez estimer."
    if found_suggestion:
        help_text += f"\n*Le système a suggéré : **{columns[default_index]}** *"

    target = st.selectbox("Colonne à prédire :", columns, index=default_index, help=help_text)
    st.session_state.target_column = target

    if st.session_state.get('shared_dataset') is not None:
        col_data = st.session_state.shared_dataset[target]
        st.info(f"**Aperçu :** Cette colonne contient des **{col_data.nunique()}** valeurs différentes.")
        with st.expander("Voir un échantillon des valeurs"):
            st.write(col_data.head(10).to_string(index=False))

    st.markdown("---")

    if st.button("Suivant →", type="primary", use_container_width=True):
        st.session_state.prediction_step = 2
        st.rerun()


def show_step2_confirmation_and_detection():
    target = st.session_state.target_column
    dataset = st.session_state.shared_dataset

    detected_type = "unknown"
    explanation = ""

    col_data = dataset[target]

    if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
        detected_type = "classification"
        explanation = "La colonne contient du texte ou des catégories (ex: Oui/Non, Client A/B)."
    elif pd.api.types.is_numeric_dtype(col_data):
        unique_count = col_data.nunique()
        total_count = len(col_data)

        if unique_count < 10 or (unique_count / total_count) < 0.05:
            detected_type = "classification"
            explanation = f"La colonne contient des nombres, mais seulement {unique_count} valeurs différentes."
        else:
            detected_type = "regression"
            explanation = "La colonne contient un grand nombre de valeurs numériques continues."

    st.session_state.problem_type = detected_type

    # Detection result card
    if detected_type == "regression":
        icon, label, color, desc = "📈", "Prédiction de Valeur (Régression)", "#2563eb", "Le modèle estimera un **nombre précis**."
    elif detected_type == "classification":
        icon, label, color, desc = "🏷️", "Prédiction de Catégorie (Classification)", "#7c3aed", "Le modèle classera les données dans un **groupe**."
    else:
        icon, label, color, desc = "❓", "Type indéterminé", "#f59e0b", "Veuillez choisir manuellement."

    st.markdown(
        f'<div style="background:white;border-radius:14px;padding:28px;'
        f'border:1px solid #e2e8f0;border-left:4px solid {color};'
        f'box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:20px;">'
        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">'
        f'<span style="font-size:32px;">{icon}</span>'
        f'<div>'
        f'<div style="font-size:11px;color:#64748b;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">'
        f'Type détecté automatiquement'
        f'</div>'
        f'<div style="font-size:17px;font-weight:700;color:{color};">{label}</div>'
        f'</div>'
        f'</div>'
        f'<div style="background:{color}0d;border-radius:8px;padding:12px 16px;'
        f'font-size:13px;color:#374151;margin-bottom:12px;">'
        f'<strong>Raison :</strong> {explanation}'
        f'</div>'
        f'<div style="font-size:13px;color:#64748b;">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Target column info
    st.markdown(
        f'<div style="background:#f8fafc;border-radius:10px;padding:14px 18px;'
        f'border:1px solid #e2e8f0;margin-bottom:20px;'
        f'display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:16px;">🎯</span>'
        f'<span style="font-size:13px;color:#374151;">'
        f'Variable cible : <strong style="color:#0f172a;font-family:monospace;">{target}</strong>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.expander("⚙️ Mode expert — forcer le type"):
        manual_type = st.radio(
            "Si la détection est incorrecte, choisissez ici :",
            ["Auto (Recommandé)", "Régression (Nombre)", "Classification (Catégorie)"],
            horizontal=True
        )
        if manual_type != "Auto (Recommandé)":
            if "Régression" in manual_type:
                st.session_state.problem_type = "regression"
            else:
                st.session_state.problem_type = "classification"

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Retour", use_container_width=True):
            st.session_state.prediction_step = 1
            st.session_state.pop('training_result', None)
            st.session_state.pop('training_in_progress', None)
            st.session_state.pop('training_start_time', None)
            st.rerun()
    with col2:
        if st.button("🚀 Lancer l'entraînement", type="primary", use_container_width=True):
            st.session_state.prediction_step = 3
            st.rerun()


def show_step3_training():
    st.markdown(
        '<div style="background:white;border-radius:14px;padding:24px;'
        'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);'
        'margin-bottom:20px;">'
        '<div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">'
        'Entraînement des modèles'
        '</div>'
        '<div style="font-size:13px;color:#64748b;">'
        'Plusieurs algorithmes sont comparés automatiquement pour trouver le plus performant'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    TIMEOUT_SECONDS = 180  # 3 minutes max

    # ── État 1 : résultat déjà disponible ────────────────────────────────────
    if 'training_result' in st.session_state:
        _render_training_result(st.session_state.training_result)
        return

    # ── État 2 : entraînement en cours — polling non-bloquant ────────────────
    if st.session_state.get('training_in_progress'):
        elapsed = time.time() - st.session_state.get('training_start_time', time.time())

        # Indicateur de progression
        dots = "." * (int(elapsed) % 4)
        st.markdown(
            f'<div style="background:#eff6ff;border-radius:10px;padding:16px 20px;'
            f'border:1px solid #bfdbfe;display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:20px;">⚙️</span>'
            f'<div>'
            f'<div style="font-size:13px;font-weight:600;color:#1d4ed8;">'
            f'Entraînement en cours{dots}</div>'
            f'<div style="font-size:12px;color:#3b82f6;margin-top:2px;">'
            f'RandomForest · GradientBoosting · LinearModel — {int(elapsed)}s écoulées'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(min(elapsed / TIMEOUT_SECONDS, 0.99))

        # Vérifier si l'agent a répondu
        response = st.session_state.bus.receive_message("Frontend")
        if response:
            st.session_state.training_result = response
            st.session_state.pop('training_in_progress', None)
            st.session_state.pop('training_start_time', None)
            st.rerun()

        # Timeout dépassé
        if elapsed > TIMEOUT_SECONDS:
            st.session_state.pop('training_in_progress', None)
            st.session_state.pop('training_start_time', None)
            st.error("Délai dépassé. L'agent ne répond pas. Vérifiez que le système multi-agent est bien démarré.")
            if st.button("↩ Réessayer", key="retry_timeout"):
                st.session_state.prediction_step = 1
                st.rerun()
            return

        # Pas encore de réponse — rerun après un court délai
        time.sleep(0.5)
        st.rerun()
        return

    # ── État 3 : démarrer l'entraînement ─────────────────────────────────────
    dataset = (st.session_state['clean_dataset']
               if 'clean_dataset' in st.session_state
               else st.session_state.get('shared_dataset'))

    msg = Message(
        sender="Frontend",
        receiver="ModelisateurML",
        message_type=MessageType.TASK_REQUEST,
        content={
            'task': 'entrainer',
            'dataset': dataset,
            'target': st.session_state.target_column,
            'problem_type': st.session_state.problem_type
        }
    )

    st.session_state.bus.send_message(msg)
    st.session_state.training_in_progress = True
    st.session_state.training_start_time = time.time()
    st.rerun()


def _render_training_result(response):
    """Affiche les résultats d'entraînement (succès ou erreur)."""
    if response.message_type == MessageType.TASK_RESPONSE:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);'
            'border:1px solid #86efac;border-left:4px solid #22c55e;'
            'border-radius:14px;padding:18px 24px;margin-bottom:20px;'
            'display:flex;align-items:center;gap:14px;">'
            '<span style="font-size:28px;">✅</span>'
            '<div>'
            '<div style="font-weight:700;color:#15803d;font-size:15px;">Entraînement terminé</div>'
            '<div style="color:#166534;font-size:13px;margin-top:2px;">'
            'Le meilleur modèle a été sélectionné'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(response.content.get('message', ''))

        model_id = response.content.get('model_id')
        if model_id:
            st.markdown(
                f'<div style="background:#f8fafc;border-radius:10px;padding:14px 18px;'
                f'border:1px solid #e2e8f0;margin:16px 0;'
                f'display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:16px;">💾</span>'
                f'<span style="font-size:13px;color:#374151;">'
                f'Modèle sauvegardé : <strong style="color:#0f172a;font-family:monospace;">{model_id}</strong>'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.button("→ Aller à la page Prédiction", type="primary", use_container_width=True):
                st.session_state.current_page = "deploy"
                st.rerun()

        with st.expander("🔍 Données techniques brutes"):
            st.json(response.content.get('results'))

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Nouvelle analyse", use_container_width=True):
                st.session_state.prediction_step = 1
                st.session_state.pop('training_result', None)
                st.rerun()
        with col2:
            if st.button("📊 Explorer les données", use_container_width=True):
                st.session_state.current_page = "explore"
                st.session_state.current_page_display = "Explorer"
                st.rerun()
        with col3:
            if st.button("🏠 Accueil", use_container_width=True):
                st.session_state.current_page = "home"
                st.session_state.current_page_display = "Accueil"
                st.rerun()

    elif response.message_type == MessageType.ERROR:
        st.error("### Une erreur est survenue")
        st.error(response.content.get('error', 'Erreur inconnue'))
        if st.button("↩ Réessayer"):
            st.session_state.prediction_step = 1
            st.session_state.pop('training_result', None)
            st.rerun()
