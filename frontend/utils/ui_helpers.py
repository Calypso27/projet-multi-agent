"""Composants UI réutilisables — thème bleu professionnel"""
import streamlit as st


# ─── Palette de couleurs ───────────────────────────────────────────────────────
COLORS = {
    "primary":       "#1d4ed8",
    "primary_dark":  "#1e3a5f",
    "primary_light": "#dbeafe",
    "accent":        "#60a5fa",
    "success":       "#22c55e",
    "warning":       "#f59e0b",
    "error":         "#ef4444",
    "bg":            "#f0f4f8",
    "card":          "#ffffff",
    "text":          "#0f172a",
    "text_muted":    "#64748b",
}


def page_header(icon: str, title: str, subtitle: str, badge: str = None, badge_color: str = None):
    """Bandeau d'en-tête de page avec icône, titre et sous-titre."""
    badge_html = ""
    if badge:
        color = badge_color or COLORS["primary"]
        badge_html = (f'<span style="background:{color}18;color:{color};border:1px solid {color}40;'
                      f'border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600;'
                      f'margin-left:auto;">{badge}</span>')

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{COLORS["primary"]}12 0%,{COLORS["primary"]}06 100%);'
        f'border:1px solid {COLORS["primary"]}22;border-left:4px solid {COLORS["primary"]};'
        f'border-radius:14px;padding:20px 28px;margin-bottom:28px;display:flex;align-items:center;gap:16px;">'
        f'<div style="background:{COLORS["primary"]}15;border-radius:12px;width:52px;height:52px;'
        f'display:flex;align-items:center;justify-content:center;font-size:26px;flex-shrink:0;">{icon}</div>'
        f'<div style="flex:1;">'
        f'<h2 style="margin:0;color:{COLORS["text"]};font-size:20px;font-weight:700;line-height:1.2;">{title}</h2>'
        f'<p style="margin:4px 0 0 0;color:{COLORS["text_muted"]};font-size:14px;">{subtitle}</p>'
        f'</div>{badge_html}</div>',
        unsafe_allow_html=True
    )


def metric_card(label: str, value: str, icon: str = "", delta: str = None,
                delta_positive: bool = True, color: str = None):
    """Card métrique stylée avec icône et delta."""
    color = color or COLORS["primary"]
    delta_html = ""
    if delta:
        delta_color = "#22c55e" if delta_positive else "#ef4444"
        delta_icon  = "↑" if delta_positive else "↓"
        delta_html  = f'<div style="font-size: 12px; color: {delta_color}; font-weight: 600; margin-top: 4px;">{delta_icon} {delta}</div>'

    st.markdown(
        f'<div style="background:white;border-radius:14px;padding:20px;border:1px solid #e2e8f0;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.06);transition:box-shadow 0.2s;height:100%;">'
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;">'
        f'<div>'
        f'<div style="font-size:12px;color:{COLORS["text_muted"]};font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:26px;font-weight:800;color:{COLORS["text"]};line-height:1;">{value}</div>'
        f'{delta_html}'
        f'</div>'
        f'<div style="background:{color}15;border-radius:10px;width:42px;height:42px;'
        f'display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">{icon}</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )


def quality_badge(score: int) -> str:
    """Retourne un badge HTML coloré selon le score qualité."""
    if score >= 90:
        color, label = "#22c55e", "Excellent"
    elif score >= 70:
        color, label = "#f59e0b", "Bon"
    elif score >= 50:
        color, label = "#f97316", "Moyen"
    else:
        color, label = "#ef4444", "Faible"

    return (f'<span style="background:{color}18;color:{color};border:1px solid {color}40;'
            f'border-radius:20px;padding:3px 10px;font-size:12px;font-weight:600;">'
            f'{label} {score}/100</span>')


def section_title(title: str, subtitle: str = ""):
    """Titre de section avec séparateur."""
    sub = (f'<p style="margin:2px 0 0 0;color:{COLORS["text_muted"]};font-size:13px;">{subtitle}</p>'
           if subtitle else "")
    st.markdown(
        f'<div style="margin:28px 0 16px 0;">'
        f'<h3 style="margin:0;color:{COLORS["text"]};font-size:16px;font-weight:700;">{title}</h3>'
        f'{sub}'
        f'<div style="height:2px;background:linear-gradient(90deg,{COLORS["primary"]},transparent);'
        f'border-radius:2px;margin-top:10px;"></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def info_card(title: str, content: str, icon: str = "ℹ️", color: str = None):
    """Card d'information stylée."""
    color = color or COLORS["primary"]
    st.markdown(
        f'<div style="background:{color}0d;border:1px solid {color}30;border-radius:12px;'
        f'padding:16px 20px;margin:8px 0;display:flex;gap:14px;align-items:flex-start;">'
        f'<span style="font-size:20px;flex-shrink:0;">{icon}</span>'
        f'<div>'
        f'<div style="font-weight:600;color:{color};font-size:14px;margin-bottom:4px;">{title}</div>'
        f'<div style="color:#374151;font-size:13px;line-height:1.5;">{content}</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )


def step_indicator(current: int, total: int, labels: list):
    """Indicateur d'étapes horizontal."""
    steps_html = '<div style="display: flex; align-items: center; margin: 16px 0 28px 0; gap: 0;">'
    for i, label in enumerate(labels):
        num = i + 1
        if num < current:
            # Complété
            bg, color, border = "#1d4ed8", "white", "#1d4ed8"
            icon = "✓"
            text_color = "#1d4ed8"
        elif num == current:
            # Actuel
            bg, color, border = "#1d4ed8", "white", "#1d4ed8"
            icon = str(num)
            text_color = "#1d4ed8"
        else:
            # À venir
            bg, color, border = "white", "#94a3b8", "#e2e8f0"
            icon = str(num)
            text_color = "#94a3b8"

        shadow = "box-shadow:0 0 0 4px #1d4ed820;" if num == current else ""
        steps_html += (
            f'<div style="display:flex;flex-direction:column;align-items:center;flex:1;">'
            f'<div style="width:34px;height:34px;background:{bg};color:{color};'
            f'border:2px solid {border};border-radius:50%;display:flex;align-items:center;'
            f'justify-content:center;font-weight:700;font-size:14px;{shadow}">{icon}</div>'
            f'<div style="font-size:11px;color:{text_color};font-weight:600;'
            f'margin-top:6px;text-align:center;">{label}</div>'
            f'</div>'
        )

        if i < len(labels) - 1:
            line_color = "#1d4ed8" if num < current else "#e2e8f0"
            steps_html += f'<div style="height: 2px; flex: 0.5; background: {line_color}; margin-bottom: 20px;"></div>'

    steps_html += "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)


def agent_status_sidebar(agents_running: bool = True):
    """Panel statut des agents dans la sidebar."""
    agents = [
        ("🎯", "Chef de Projet"),
        ("⚙️", "Ingénieur Données"),
        ("🔍", "Analyste"),
        ("🤖", "Modélisateur ML"),
    ]
    dot_color = "#22c55e" if agents_running else "#94a3b8"
    dot_shadow = f"0 0 6px {dot_color}" if agents_running else "none"
    status_label = "Actif" if agents_running else "En attente"

    html = (
        f'<div style="margin:4px 0;">'
        f'<div style="font-size:10px;color:#475569;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">'
        f'Agents — <span style="color:{dot_color};">{status_label}</span></div>'
    )
    for icon, name in agents:
        html += (
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'padding:7px 10px;margin:3px 0;border-radius:8px;'
            f'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);">'
            f'<span style="font-size:14px;">{icon}</span>'
            f'<span style="font-size:12px;color:#94a3b8;font-weight:500;flex:1;">{name}</span>'
            f'<span style="width:7px;height:7px;background:{dot_color};'
            f'border-radius:50%;box-shadow:{dot_shadow};"></span>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
