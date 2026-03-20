"""Page de prediction"""
import streamlit as st
import pandas as pd
import numpy as np
from backend.models.model_manager import ModelManager
from frontend.utils.ui_helpers import page_header
import io


def render_deploy():
    page_header("🚀", "Prédiction",
                "Utilisez vos modèles entraînés pour faire des prédictions",
                badge="Étape 4/5")

    models = ModelManager.list_models()

    if not models:
        st.markdown("""<div style="background:#fffbeb;border-radius:14px;padding:28px;"""
                    """border:1px solid #fde68a;border-left:4px solid #f59e0b;"""
                    """text-align:center;margin:20px 0;">"""
                    """<div style="font-size:40px;margin-bottom:12px;">🤖</div>"""
                    """<div style="font-size:16px;font-weight:700;color:#92400e;margin-bottom:8px;">"""
                    """Aucun modèle disponible</div>"""
                    """<div style="font-size:13px;color:#78350f;">"""
                    """Entraînez d'abord un modèle dans l'étape Modélisation</div>"""
                    """</div>""", unsafe_allow_html=True)

        if st.button("→ Aller à la modélisation", type="primary", use_container_width=True):
            st.session_state.current_page = "predict"
            st.session_state.current_page_display = "Prédire"
            st.rerun()
        return

    # === ÉTAPE 1 : Sélection du modèle ===
    st.markdown("""<div style="background:white;border-radius:14px;padding:24px;"""
                """border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:20px;">"""
                """<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px;">"""
                """Étape 1 — Sélectionner un modèle</div>"""
                """<div style="font-size:13px;color:#64748b;">Choisissez parmi les modèles entraînés</div>"""
                """</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        models_df = pd.DataFrame([{
            'Nom': m['name'],
            'Type': m['type'],
            'Cible': m['target'],
            'Créé le': pd.to_datetime(m['created_at']).strftime('%Y-%m-%d %H:%M'),
            'Features': m['n_features'],
            'Performance': _format_metrics(m['metrics'], m['type'])
        } for m in models])

        st.dataframe(models_df, use_container_width=True, hide_index=True)

        model_options = [f"{m['name']} — {m['target']} ({m['created_at'][:10]})" for m in models]
        selected_idx = st.selectbox(
            "Choisir le modèle à utiliser :",
            range(len(model_options)),
            format_func=lambda i: model_options[i]
        )

        selected_model_info = models[selected_idx]

    with col2:
        metrics = selected_model_info['metrics']
        problem_type = selected_model_info['type']
        perf_label = _format_metrics(metrics, problem_type)
        type_icon = "📈" if problem_type == "regression" else "🏷️"
        type_color = "#2563eb" if problem_type == "regression" else "#7c3aed"

        st.markdown(
            f'<div style="background:white;border-radius:14px;padding:20px;'
            f'border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
            f'<div style="font-size:12px;color:#64748b;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px;">Détails du modèle</div>'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
            f'<span style="font-size:20px;">{type_icon}</span>'
            f'<span style="font-size:13px;font-weight:700;color:{type_color};">'
            f'{problem_type.capitalize()}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:#64748b;margin-bottom:4px;">Variable cible</div>'
            f'<div style="font-size:13px;font-weight:600;color:#0f172a;'
            f'font-family:monospace;margin-bottom:12px;">{selected_model_info["target"]}</div>'
            f'<div style="font-size:12px;color:#64748b;margin-bottom:4px;">Performance</div>'
            f'<div style="font-size:13px;font-weight:700;color:#22c55e;margin-bottom:12px;">'
            f'{perf_label}</div>'
            f'<div style="font-size:12px;color:#64748b;margin-bottom:4px;">Features</div>'
            f'<div style="font-size:13px;font-weight:600;color:#0f172a;">'
            f'{selected_model_info["n_features"]} variables</div>'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    model, metadata = ModelManager.load_model(selected_model_info['id'])

    # === ÉTAPE 2 : Saisie des données ===
    st.markdown("""<div style="background:white;border-radius:14px;padding:24px;"""
                """border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:20px;">"""
                """<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px;">"""
                """Étape 2 — Entrer les données</div>"""
                """<div style="font-size:13px;color:#64748b;">"""
                """Formulaire pour une prédiction ou fichier pour plusieurs</div>"""
                """</div>""", unsafe_allow_html=True)

    input_mode = st.radio(
        "Mode de saisie :",
        ["📝 Formulaire (prédiction unique)", "📁 Fichier (prédictions multiples)"],
        help="Formulaire pour une prédiction rapide, fichier pour prédictions en masse"
    )

    new_data = None
    is_single_prediction = False

    if "Formulaire" in input_mode:
        is_single_prediction = True
        new_data = _render_prediction_form(metadata)
    else:
        new_data = _render_file_upload(selected_model_info)

    st.markdown("<br>", unsafe_allow_html=True)

    # === ÉTAPE 3 : Lancer la prédiction ===
    st.markdown("""<div style="background:white;border-radius:14px;padding:24px;"""
                """border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:20px;">"""
                """<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px;">"""
                """Étape 3 — Générer les prédictions</div>"""
                """<div style="font-size:13px;color:#64748b;">Cliquez pour lancer l'inférence</div>"""
                """</div>""", unsafe_allow_html=True)

    if new_data is not None:
        button_label = "🔮 Prédire" if is_single_prediction else "🔮 Lancer les prédictions"

        if st.button(button_label, type="primary", use_container_width=True):
            with st.spinner("Génération des prédictions en cours..."):
                try:
                    X_pred = _prepare_prediction_data(new_data, metadata)
                    predictions = model.predict(X_pred)

                    if 'workflow' in st.session_state:
                        st.session_state.workflow['step_4_model_deployed'] = True

                    if is_single_prediction:
                        _display_single_prediction(predictions[0], metadata)
                    else:
                        _display_multiple_predictions(new_data, predictions, metadata, selected_model_info)

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {str(e)}")
                    import traceback
                    with st.expander("Détails de l'erreur"):
                        st.code(traceback.format_exc())
    else:
        st.markdown("""<div style="background:#f8fafc;border-radius:10px;padding:14px 18px;"""
                    """border:1px solid #e2e8f0;font-size:13px;color:#64748b;">"""
                    """💡 Remplissez le formulaire ou chargez un fichier pour activer la prédiction."""
                    """</div>""", unsafe_allow_html=True)


def _render_prediction_form(metadata: dict) -> pd.DataFrame:
    dataset = st.session_state.get('shared_dataset')
    feature_names = metadata.get('feature_names', [])
    target_column = metadata.get('target_column', '')

    if dataset is None:
        st.warning("Dataset original non disponible. Les valeurs par défaut seront utilisées.")
        dataset = pd.DataFrame()

    form_values = {}

    original_columns = set()
    for feat in feature_names:
        if '_' in feat:
            parts = feat.rsplit('_', 1)
            if len(parts) == 2:
                original_columns.add(parts[0])
        else:
            original_columns.add(feat)

    if not dataset.empty:
        columns_to_use = [col for col in dataset.columns if col != target_column]
    else:
        columns_to_use = list(original_columns)

    st.markdown("""<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:12px;">"""
                """Renseignez les variables de prédiction</div>""", unsafe_allow_html=True)

    n_cols = 2
    cols = st.columns(n_cols)

    for idx, col_name in enumerate(columns_to_use):
        with cols[idx % n_cols]:
            if not dataset.empty and col_name in dataset.columns:
                col_data = dataset[col_name]

                if pd.api.types.is_numeric_dtype(col_data):
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())

                    if pd.api.types.is_integer_dtype(col_data):
                        form_values[col_name] = st.number_input(
                            f"{col_name}",
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(mean_val),
                            help=f"Valeur entre {int(min_val)} et {int(max_val)}"
                        )
                    else:
                        form_values[col_name] = st.number_input(
                            f"{col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            format="%.2f",
                            help=f"Valeur entre {min_val:.2f} et {max_val:.2f}"
                        )

                elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                    unique_values = col_data.dropna().unique().tolist()
                    form_values[col_name] = st.selectbox(
                        f"{col_name}",
                        options=unique_values,
                        help=f"{len(unique_values)} valeurs possibles"
                    )

                elif pd.api.types.is_bool_dtype(col_data):
                    form_values[col_name] = st.checkbox(f"{col_name}", value=False)

                else:
                    form_values[col_name] = st.text_input(f"{col_name}", value="")

            else:
                form_values[col_name] = st.text_input(f"{col_name}", value="")

    with st.expander("Résumé des valeurs saisies"):
        summary_df = pd.DataFrame([form_values])
        st.dataframe(summary_df, use_container_width=True)

    return pd.DataFrame([form_values])


def _render_file_upload(selected_model_info: dict) -> pd.DataFrame:
    data_source = st.radio(
        "Source des données :",
        ["Utiliser le dataset actuel", "Uploader un nouveau fichier"],
        help="Utilisez le dataset déjà chargé ou uploadez de nouvelles données"
    )

    new_data = None

    if data_source == "Utiliser le dataset actuel":
        if st.session_state.get('shared_dataset') is not None:
            dataset = st.session_state.shared_dataset
            target = selected_model_info['target']

            if target in dataset.columns:
                new_data = dataset.drop(columns=[target])
            else:
                new_data = dataset.copy()

            st.markdown(
                f'<div style="background:#f0fdf4;border-radius:10px;padding:12px 16px;'
                f'border:1px solid #bbf7d0;font-size:13px;color:#15803d;'
                f'display:flex;align-items:center;gap:8px;">'
                f'<span>✅</span>'
                f'<span>Dataset chargé : <strong>{new_data.shape[0]:,} lignes × {new_data.shape[1]} colonnes</strong></span>'
                f'</div>',
                unsafe_allow_html=True)

            with st.expander("Aperçu des données"):
                st.dataframe(new_data.head(10))
        else:
            st.warning("Aucun dataset chargé.")

    else:
        uploaded_file = st.file_uploader(
            "Uploader un fichier pour prédiction",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Le fichier doit contenir les mêmes colonnes que lors de l'entraînement"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    new_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    new_data = pd.read_json(uploaded_file)

                st.success(f"Fichier chargé : {new_data.shape[0]:,} lignes × {new_data.shape[1]} colonnes")

                with st.expander("Aperçu des données"):
                    st.dataframe(new_data.head(10))

            except Exception as e:
                st.error(f"Erreur lors du chargement : {str(e)}")

    return new_data


def _display_single_prediction(prediction, metadata: dict):
    target_column = metadata.get('target_column', 'Résultat')
    problem_type = metadata.get('problem_type', 'classification')

    if problem_type == 'classification':
        gradient = "linear-gradient(135deg,#1d4ed8,#2563eb)"
        value_display = str(prediction)
    else:
        gradient = "linear-gradient(135deg,#0891b2,#0e7490)"
        try:
            value_display = f"{prediction:.4f}"
        except Exception:
            value_display = str(prediction)

    type_label = "Classe prédite" if problem_type == "classification" else "Valeur estimée"
    st.markdown(
        f'<div style="background:{gradient};border-radius:16px;padding:36px 24px;'
        f'text-align:center;margin:20px 0;box-shadow:0 4px 20px rgba(37,99,235,.25);">'
        f'<div style="font-size:13px;color:rgba(255,255,255,.8);font-weight:600;'
        f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">'
        f'Prédiction — {target_column}</div>'
        f'<div style="font-size:52px;font-weight:800;color:white;'
        f'letter-spacing:-.02em;line-height:1.1;margin:16px 0;">{value_display}</div>'
        f'<div style="font-size:13px;color:rgba(255,255,255,.7);">{type_label}</div>'
        f'</div>',
        unsafe_allow_html=True)

    metrics = metadata.get('metrics', {})
    model_name = metadata.get('model_name', 'N/A')

    if problem_type == 'classification':
        perf_label = "Précision"
        perf_value = f"{metrics.get('Accuracy', 0):.1%}"
    else:
        perf_label = "Score R²"
        perf_value = f"{metrics.get('R²', 0):.3f}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Modèle", model_name)
    with c2:
        st.metric("Type", problem_type.capitalize())
    with c3:
        st.metric(perf_label, perf_value)


def _display_multiple_predictions(new_data: pd.DataFrame, predictions, metadata: dict, selected_model_info: dict):
    result_df = new_data.copy()
    result_df[f'Prediction_{metadata["target_column"]}'] = predictions

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);'
        f'border:1px solid #86efac;border-left:4px solid #22c55e;'
        f'border-radius:14px;padding:18px 24px;margin-bottom:20px;'
        f'display:flex;align-items:center;gap:14px;">'
        f'<span style="font-size:28px;">✅</span>'
        f'<div>'
        f'<div style="font-weight:700;color:#15803d;font-size:15px;">Prédictions générées</div>'
        f'<div style="color:#166534;font-size:13px;margin-top:2px;">'
        f'{len(predictions):,} prédictions calculées avec succès</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total", f"{len(predictions):,}")
    with c2:
        if metadata['problem_type'] == 'classification':
            st.metric("Classes prédites", len(np.unique(predictions)))
        else:
            st.metric("Moyenne", f"{np.mean(predictions):.2f}")
    with c3:
        if metadata['problem_type'] == 'regression':
            st.metric("Écart-type", f"{np.std(predictions):.2f}")
        else:
            from collections import Counter
            most_common = Counter(predictions).most_common(1)[0][0]
            st.metric("Classe majoritaire", most_common)

    st.dataframe(result_df, use_container_width=True)

    st.markdown("### Distribution des prédictions")
    if metadata['problem_type'] == 'classification':
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        st.bar_chart(pred_counts)
    else:
        st.bar_chart(pd.Series(predictions).value_counts(bins=20).sort_index())

    st.markdown("### Exporter les résultats")

    col1, col2 = st.columns(2)
    with col1:
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Télécharger CSV",
            data=csv,
            file_name=f"predictions_{selected_model_info['id'][:10]}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Predictions')

        st.download_button(
            label="⬇ Télécharger Excel",
            data=buffer.getvalue(),
            file_name=f"predictions_{selected_model_info['id'][:10]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.session_state.last_predictions = result_df
    st.session_state.last_model_used = selected_model_info


def _format_metrics(metrics: dict, problem_type: str) -> str:
    if problem_type == 'regression':
        r2 = metrics.get('R²', 0)
        return f"R² = {r2:.3f}"
    else:
        acc = metrics.get('Accuracy', 0)
        return f"Acc = {acc:.3f}"


def _prepare_prediction_data(df: pd.DataFrame, metadata: dict) -> np.ndarray:
    X = df.copy()
    X = pd.get_dummies(X, drop_first=True)

    training_features = metadata.get('feature_names', [])

    for feat in training_features:
        if feat not in X.columns:
            X[feat] = 0

    X = X.reindex(columns=training_features, fill_value=0)
    X = X.fillna(X.median())
    X = X.fillna(0)

    return X.values
