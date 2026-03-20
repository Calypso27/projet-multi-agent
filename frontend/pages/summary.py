"""Page de rapport"""
import streamlit as st
from datetime import datetime
import pandas as pd
from backend.models.model_manager import ModelManager
from backend.utils.data_quality_scorer import DataQualityScorer
from frontend.utils.ui_helpers import page_header


def render_summary():
    page_header("📋", "Executive Summary",
                "Rapport synthétique de l'analyse et des modèles produits",
                badge="Étape 5/5")

    if not st.session_state.get('dataset_loaded'):
        st.markdown("""<div style="background:#fffbeb;border-radius:14px;padding:28px;"""
                    """border:1px solid #fde68a;border-left:4px solid #f59e0b;"""
                    """text-align:center;margin:20px 0;">"""
                    """<div style="font-size:40px;margin-bottom:12px;">⚠️</div>"""
                    """<div style="font-size:15px;font-weight:700;color:#92400e;margin-bottom:8px;">"""
                    """Aucune donnée chargée</div>"""
                    """<div style="font-size:13px;color:#78350f;">"""
                    """Commencez par charger un dataset pour générer un rapport.</div>"""
                    """</div>""", unsafe_allow_html=True)
        if st.button("← Aller à l'accueil", type="primary"):
            st.session_state.current_page = "home"
            st.session_state.current_page_display = "Accueil"
            st.rerun()
        return

    workflow = st.session_state.get('workflow', {})
    if not workflow.get('step_2_data_explored', False):
        st.warning("Explorez d'abord vos données pour avoir un rapport complet.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Explorer les données", type="primary", use_container_width=True):
                st.session_state.current_page = "explore"
                st.session_state.current_page_display = "Explorer"
                st.rerun()
        with col2:
            if st.button("Générer le rapport quand même", use_container_width=True):
                st.session_state.workflow['step_2_data_explored'] = True
                st.rerun()
        return

    info = st.session_state.dataset_info
    profile = st.session_state.dataset_profile
    models = ModelManager.list_models()

    quality_score = info.get('quality_score', 0)
    q_color = "#22c55e" if quality_score >= 90 else "#f59e0b" if quality_score >= 70 else "#ef4444"
    q_label = "Excellent" if quality_score >= 90 else "Bon" if quality_score >= 70 else "À améliorer"

    # ─── VUE D'ENSEMBLE ────────────────────────────────────────────────────────
    st.markdown("""<div style="font-size:16px;font-weight:700;color:#0f172a;"""
                """margin:24px 0 16px;padding-left:4px;">1. Vue d'ensemble</div>""",
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        rows_html = ""
        rows_data = [
            ("📁", "Dataset", f"<code style='background:#f1f5f9;padding:2px 6px;border-radius:4px;'>{info['filename']}</code>"),
            ("📅", "Date d'analyse", datetime.now().strftime('%d/%m/%Y %H:%M')),
            ("📏", "Volume", f"{info['rows']:,} lignes × {info['columns']} colonnes"),
            ("🗂️", "Format", info['format'].upper()),
            ("🔍", "Type de données", info['data_type']),
        ]
        for icon, label, value in rows_data:
            rows_html += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:10px 0;border-bottom:1px solid #f1f5f9;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:15px;">{icon}</span>'
                f'<span style="font-size:13px;color:#64748b;">{label}</span>'
                f'</div>'
                f'<span style="font-size:13px;color:#0f172a;font-weight:600;">{value}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:white;border-radius:14px;padding:20px 24px;'
            f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True)

    with col2:
        st.markdown(
            f'<div style="background:white;border-radius:14px;padding:24px;'
            f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);text-align:center;">'
            f'<div style="font-size:11px;color:#64748b;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;">Score qualité</div>'
            f'<div style="font-size:52px;font-weight:800;color:{q_color};line-height:1;margin-bottom:8px;">'
            f'{quality_score}</div>'
            f'<div style="font-size:14px;color:#94a3b8;margin-bottom:14px;">/100</div>'
            f'<div style="background:{q_color}18;color:{q_color};border:1px solid {q_color}40;'
            f'border-radius:20px;padding:5px 14px;font-size:12px;font-weight:700;'
            f'display:inline-block;">{q_label}</div>'
            f'<div style="margin-top:16px;background:#f1f5f9;border-radius:6px;height:6px;overflow:hidden;">'
            f'<div style="width:{quality_score}%;height:100%;'
            f'background:linear-gradient(90deg,{q_color}cc,{q_color});border-radius:6px;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── KPI ───────────────────────────────────────────────────────────────────
    st.markdown("""<div style="font-size:16px;font-weight:700;color:#0f172a;"""
                """margin:8px 0 16px;padding-left:4px;">2. Indicateurs clés (KPI)</div>""",
                unsafe_allow_html=True)

    completeness = 100 - (profile['missing_values']['percentage'] if profile['missing_values']['total'] > 0 else 0)
    duplicates_pct = (profile['duplicates'] / info['rows'] * 100) if info['rows'] > 0 else 0
    quality_report = profile.get('quality_report', {})
    usable_rows = quality_report.get('summary', {}).get('usable_rows', info['rows'])
    models_count = len(models)

    kpi_data = [
        ("✅", "Complétude", f"{completeness:.1f}%", "#22c55e" if completeness >= 95 else "#f59e0b"),
        ("🔁", "Données uniques", f"{100 - duplicates_pct:.1f}%", "#22c55e" if duplicates_pct == 0 else "#f59e0b"),
        ("📊", "Lignes exploitables", f"{usable_rows:,}", "#2563eb"),
        ("🤖", "Modèles entraînés", str(models_count), "#22c55e" if models_count > 0 else "#f59e0b"),
    ]

    cols = st.columns(4)
    for i, (icon, label, value, color) in enumerate(kpi_data):
        with cols[i]:
            st.markdown(
                f'<div style="background:white;border-radius:12px;padding:20px;'
                f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);text-align:center;">'
                f'<div style="font-size:22px;margin-bottom:8px;">{icon}</div>'
                f'<div style="font-size:24px;font-weight:800;color:{color};line-height:1;margin-bottom:6px;">'
                f'{value}</div>'
                f'<div style="font-size:11px;color:#64748b;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:.05em;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── QUALITÉ DES DONNÉES ────────────────────────────────────────────────────
    st.markdown("""<div style="font-size:16px;font-weight:700;color:#0f172a;"""
                """margin:8px 0 16px;padding-left:4px;">3. Évaluation de la qualité</div>""",
                unsafe_allow_html=True)

    if 'quality_report' in profile:
        qr = profile['quality_report']
        dimensions = qr['dimensions']

        dim_names = ['Complétude', 'Validité', 'Cohérence', 'Unicité', 'Exactitude']
        dim_keys = ['completeness', 'validity', 'consistency', 'uniqueness', 'accuracy']

        dim_rows = ""
        for name, key in zip(dim_names, dim_keys):
            d = dimensions[key]
            score = d['score']
            status = d['status'].upper()
            bar_color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
            dim_rows += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:10px 0;border-bottom:1px solid #f8fafc;">'
                f'<div style="font-size:13px;color:#374151;width:120px;">{name}</div>'
                f'<div style="flex:1;margin:0 16px;background:#f1f5f9;border-radius:4px;height:6px;overflow:hidden;">'
                f'<div style="width:{score}%;height:100%;background:{bar_color};border-radius:4px;"></div>'
                f'</div>'
                f'<div style="font-size:13px;font-weight:700;color:{bar_color};width:40px;text-align:right;">'
                f'{score}</div>'
                f'<div style="font-size:11px;color:#94a3b8;width:60px;text-align:right;'
                f'font-weight:600;text-transform:uppercase;">{status}</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:white;border-radius:14px;padding:20px 24px;'
            f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:16px;">'
            f'{dim_rows}'
            f'</div>',
            unsafe_allow_html=True)

        recommendations = qr.get('recommendations', [])
        critical_recs = [r for r in recommendations if "critique" in r.lower() or "critical" in r.lower()]
        warning_recs = [r for r in recommendations if "attention" in r.lower() or "warning" in r.lower()]

        if critical_recs:
            st.error("**Actions critiques requises :**")
            for rec in critical_recs:
                st.markdown(f"- {rec}")
        if warning_recs:
            st.warning("**Améliorations recommandées :**")
            for rec in warning_recs[:3]:
                st.markdown(f"- {rec}")
        if not critical_recs and not warning_recs:
            st.success("Données de qualité optimale — aucune action requise.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── CARACTÉRISTIQUES ──────────────────────────────────────────────────────
    st.markdown("""<div style="font-size:16px;font-weight:700;color:#0f172a;"""
                """margin:8px 0 16px;padding-left:4px;">4. Caractéristiques des données</div>""",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        col_types = profile['column_types']
        type_rows = ""
        type_data = [
            ("🔢", "Colonnes numériques", col_types.get('numeric', 0), "#2563eb"),
            ("🔤", "Colonnes catégorielles", col_types.get('categorical', 0), "#7c3aed"),
            ("📅", "Colonnes dates", col_types.get('datetime', 0), "#0891b2"),
        ]
        for icon, label, val, color in type_data:
            type_rows += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:10px 0;border-bottom:1px solid #f8fafc;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="font-size:15px;">{icon}</span>'
                f'<span style="font-size:13px;color:#374151;">{label}</span>'
                f'</div>'
                f'<span style="background:{color}18;color:{color};border:1px solid {color}30;'
                f'border-radius:8px;padding:3px 10px;font-size:13px;font-weight:700;">{val}</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:white;border-radius:14px;padding:20px 24px;'
            f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
            f'<div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:12px;">Types de colonnes</div>'
            f'{type_rows}'
            f'</div>',
            unsafe_allow_html=True)

    with col2:
        missing = profile['missing_values']
        dupes = profile['duplicates']
        problematic = profile.get('quality_report', {}).get('summary', {}).get('problematic_columns', 0)

        attention_rows = ""
        attention_data = [
            ("❓", "Valeurs manquantes", f"{missing['total']:,} ({missing['percentage']:.1f}%)",
             "#f59e0b" if missing['total'] > 0 else "#22c55e"),
            ("📋", "Lignes dupliquées", str(dupes),
             "#f59e0b" if dupes > 0 else "#22c55e"),
            ("⚠️", "Colonnes problématiques", str(problematic),
             "#ef4444" if problematic > 0 else "#22c55e"),
        ]
        for icon, label, val, color in attention_data:
            attention_rows += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:10px 0;border-bottom:1px solid #f8fafc;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="font-size:15px;">{icon}</span>'
                f'<span style="font-size:13px;color:#374151;">{label}</span>'
                f'</div>'
                f'<span style="background:{color}18;color:{color};border:1px solid {color}30;'
                f'border-radius:8px;padding:3px 10px;font-size:13px;font-weight:700;">{val}</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:white;border-radius:14px;padding:20px 24px;'
            f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
            f'<div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:12px;">Points d\'attention</div>'
            f'{attention_rows}'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── MODÈLES ML ────────────────────────────────────────────────────────────
    st.markdown("""<div style="font-size:16px;font-weight:700;color:#0f172a;"""
                """margin:8px 0 16px;padding-left:4px;">5. Modèles Machine Learning</div>""",
                unsafe_allow_html=True)

    if models:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);'
            f'border:1px solid #86efac;border-left:4px solid #22c55e;'
            f'border-radius:14px;padding:14px 20px;margin-bottom:16px;'
            f'display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:20px;">🤖</span>'
            f'<span style="font-size:13px;font-weight:700;color:#15803d;">'
            f'{len(models)} modèle(s) entraîné(s) et disponible(s) pour déploiement</span>'
            f'</div>',
            unsafe_allow_html=True)

        models_df = pd.DataFrame([{
            'Modèle': m['name'],
            'Type': m['type'].capitalize(),
            'Variable cible': m['target'],
            'Performance': _format_model_performance(m['metrics'], m['type']),
            'Date création': pd.to_datetime(m['created_at']).strftime('%d/%m/%Y %H:%M'),
            'Features': m['n_features']
        } for m in models])

        st.dataframe(models_df, use_container_width=True, hide_index=True)

        best = models[0]
        st.markdown(
            f'<div style="background:#eff6ff;border-radius:12px;padding:16px 20px;'
            f'border:1px solid #bfdbfe;margin-top:12px;">'
            f'<div style="font-size:12px;color:#1d4ed8;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">'
            f'Modèle recommandé pour le déploiement</div>'
            f'<div style="font-size:14px;font-weight:700;color:#1e40af;">{best["name"]}</div>'
            f'<div style="font-size:12px;color:#3b82f6;margin-top:4px;">'
            f'{_format_model_performance(best["metrics"], best["type"])} · Cible : {best["target"]}</div>'
            f'</div>',
            unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#f8fafc;border-radius:14px;padding:24px;"""
                    """border:1px solid #e2e8f0;text-align:center;">"""
                    """<div style="font-size:32px;margin-bottom:8px;">🤖</div>"""
                    """<div style="font-size:14px;color:#64748b;">Aucun modèle entraîné pour le moment.</div>"""
                    """</div>""", unsafe_allow_html=True)
        if st.button("→ Entraîner un modèle", type="primary"):
            st.session_state.current_page = "predict"
            st.session_state.current_page_display = "Prédire"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── PROCHAINES ÉTAPES ─────────────────────────────────────────────────────
    st.markdown("""<div style="font-size:16px;font-weight:700;color:#0f172a;"""
                """margin:8px 0 16px;padding-left:4px;">6. Prochaines étapes recommandées</div>""",
                unsafe_allow_html=True)

    has_models = len(models) > 0
    quality_ok = quality_score >= 70

    next_steps = []
    if not quality_ok:
        next_steps.append(("🔴", "CRITIQUE", "Améliorer la qualité des données",
                           "Traiter les valeurs manquantes et incohérences détectées"))
    if not has_models:
        next_steps.append(("🔴", "CRITIQUE", "Entraîner un modèle ML",
                           "Créer un modèle prédictif pour exploiter les données"))
    if has_models and quality_ok:
        next_steps.append(("🟢", "RECOMMANDÉ", "Déployer le modèle en production",
                           "Utiliser le modèle pour faire des prédictions sur de nouvelles données"))
    next_steps.append(("🔵", "OPTIONNEL", "Approfondir l'analyse exploratoire",
                       "Explorer les corrélations et patterns dans les données"))
    next_steps.append(("🔵", "OPTIONNEL", "Optimiser les hyperparamètres",
                       "Améliorer les performances des modèles existants"))

    priority_color = {"CRITIQUE": "#ef4444", "RECOMMANDÉ": "#22c55e", "OPTIONNEL": "#2563eb"}
    steps_rows = ""
    for dot, priority, action, desc in next_steps:
        color = priority_color.get(priority, "#64748b")
        steps_rows += (
            f'<div style="display:flex;align-items:flex-start;gap:14px;'
            f'padding:12px 0;border-bottom:1px solid #f1f5f9;">'
            f'<span style="font-size:18px;margin-top:2px;">{dot}</span>'
            f'<div style="flex:1;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            f'<span style="font-size:12px;font-weight:700;color:{color};'
            f'background:{color}14;border:1px solid {color}30;'
            f'border-radius:12px;padding:2px 10px;">{priority}</span>'
            f'<span style="font-size:13px;font-weight:600;color:#0f172a;">{action}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:#64748b;">{desc}</div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:white;border-radius:14px;padding:20px 24px;'
        f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:20px;">'
        f'{steps_rows}'
        f'</div>',
        unsafe_allow_html=True)

    # ─── EXPORT ────────────────────────────────────────────────────────────────
    st.markdown("""<div style="font-size:16px;font-weight:700;color:#0f172a;"""
                """margin:8px 0 16px;padding-left:4px;">7. Export du rapport</div>""",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        full_report = _generate_full_markdown_report(info, profile, models)
        st.download_button(
            label="⬇ Rapport Markdown",
            data=full_report,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        if 'quality_report' in profile:
            quality_md = DataQualityScorer.format_report(profile['quality_report'])
            st.download_button(
                label="⬇ Rapport qualité",
                data=quality_md,
                file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    with col3:
        import json
        import numpy as np

        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        export_data = {
            'dataset_info': convert_numpy_types(info),
            'quality_score': float(info.get('quality_score', 0)) if info.get('quality_score') else None,
            'models_count': len(models),
            'generated_at': datetime.now().isoformat()
        }

        st.download_button(
            label="⬇ Données JSON",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"summary_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )

    # ─── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;padding:24px;'
        f'background:linear-gradient(135deg,#0f172a,#1a2744);border-radius:14px;margin-top:32px;">'
        f'<div style="color:#60a5fa;font-size:14px;font-weight:600;margin-bottom:6px;">'
        f'📊 DataAgent — Plateforme ML Multi-Agent</div>'
        f'<div style="color:#64748b;font-size:12px;">'
        f'Rapport généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}</div>'
        f'</div>',
        unsafe_allow_html=True)


def _format_model_performance(metrics: dict, problem_type: str) -> str:
    if problem_type == 'regression':
        r2 = metrics.get('R²', 0)
        return f"R² = {r2:.3f}"
    else:
        acc = metrics.get('Accuracy', 0)
        return f"Accuracy = {acc:.3f}"


def _generate_full_markdown_report(info: dict, profile: dict, models: list) -> str:
    report = f"""# Executive Summary — Analyse de Données

**Dataset :** {info['filename']}
**Date :** {datetime.now().strftime('%d/%m/%Y %H:%M')}

---

## 1. Vue d'ensemble

- **Volume :** {info['rows']:,} lignes × {info['columns']} colonnes
- **Format :** {info['format'].upper()}
- **Type :** {info['data_type']}
- **Qualité globale :** {info.get('quality_score', 'N/A')}/100 ({info.get('quality_grade', 'N/A')})

## 2. Indicateurs clés

| Métrique | Valeur |
|----------|--------|
| Complétude | {100 - profile['missing_values']['percentage']:.1f}% |
| Valeurs manquantes | {profile['missing_values']['total']:,} |
| Lignes dupliquées | {profile['duplicates']} |
| Colonnes numériques | {profile['column_types']['numeric']} |
| Colonnes catégorielles | {profile['column_types']['categorical']} |

## 3. Qualité des données

"""

    if 'quality_report' in profile:
        quality_report = profile['quality_report']
        dimensions = quality_report['dimensions']

        report += "| Dimension | Score | Statut |\n"
        report += "|-----------|-------|--------|\n"

        dim_names = {
            'completeness': 'Complétude',
            'validity': 'Validité',
            'consistency': 'Cohérence',
            'uniqueness': 'Unicité',
            'accuracy': 'Exactitude'
        }

        for key, name in dim_names.items():
            dim = dimensions[key]
            report += f"| {name} | {dim['score']}/100 | {dim['status'].upper()} |\n"

        report += "\n### Recommandations\n\n"
        for rec in quality_report.get('recommendations', []):
            report += f"- {rec}\n"

    report += "\n## 4. Modèles Machine Learning\n\n"

    if models:
        report += f"**{len(models)} modèle(s) entraîné(s)**\n\n"
        report += "| Modèle | Type | Variable | Performance |\n"
        report += "|--------|------|----------|-------------|\n"

        for m in models:
            perf = _format_model_performance(m['metrics'], m['type'])
            report += f"| {m['name']} | {m['type']} | {m['target']} | {perf} |\n"
    else:
        report += "*Aucun modèle entraîné pour le moment.*\n"

    report += f"\n---\n\n*Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n"

    return report
