"""Page de prediction"""
import streamlit as st
import pandas as pd
import numpy as np
from backend.models.model_manager import ModelManager
import io


def render_deploy():
    st.markdown("# Prédiction")
    st.markdown("*Utilisez vos modèles pour faire des prédictions*")

    models = ModelManager.list_models()

    if not models:
        st.warning("**Étape précédente requise**")
        st.info("Vous devez d'abord entraîner un modèle avant de pouvoir faire des prédictions.")

        if st.button("Aller à la modélisation", type="primary", use_container_width=True):
            st.session_state.current_page = "predict"
            st.session_state.current_page_display = "Prédire"
            st.rerun()
        return

    st.markdown("## Étape 1: Sélectionner un modèle")

    col1, col2 = st.columns([2, 1])

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

        model_options = [f"{m['name']} - {m['target']} ({m['created_at'][:10]})" for m in models]
        selected_idx = st.selectbox(
            "Choisir le modèle à utiliser:",
            range(len(model_options)),
            format_func=lambda i: model_options[i]
        )

        selected_model_info = models[selected_idx]

    with col2:
        st.markdown("### Détails du modèle")
        st.markdown(f"**ID:** `{selected_model_info['id'][:20]}...`")
        st.markdown(f"**Type:** {selected_model_info['type'].capitalize()}")
        st.markdown(f"**Variable cible:** {selected_model_info['target']}")
        st.markdown(f"**Nombre de features:** {selected_model_info['n_features']}")

        metrics = selected_model_info['metrics']
        st.markdown("**Performance:**")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                st.markdown(f"- {key}: {value:.4f}")

    st.markdown("---")

    model, metadata = ModelManager.load_model(selected_model_info['id'])
    feature_names = metadata.get('feature_names', [])

    st.markdown("## Étape 2: Entrer les données")

    input_mode = st.radio(
        "Mode de saisie:",
        ["Formulaire (prédiction unique)", "Fichier (prédictions multiples)"],
        help="Choisissez le formulaire pour une prédiction rapide ou uploadez un fichier pour plusieurs prédictions"
    )

    new_data = None
    is_single_prediction = False

    if input_mode == "Formulaire (prédiction unique)":
        is_single_prediction = True
        new_data = _render_prediction_form(metadata)

    else:  # Mode fichier
        new_data = _render_file_upload(selected_model_info)

    st.markdown("---")
    st.markdown("## Étape 3: Générer les prédictions")

    if new_data is not None:
        button_label = "Prédire" if is_single_prediction else "Lancer les prédictions"

        if st.button(button_label, type="primary", use_container_width=True):
            with st.spinner("Génération des prédictions en cours..."):
                try:
                    X_pred = _prepare_prediction_data(new_data, metadata)
                    predictions = model.predict(X_pred)

                    if 'workflow' in st.session_state:
                        st.session_state.workflow['step_4_model_deployed'] = True

                    if is_single_prediction:
                        # Affichage pour prédiction unique
                        _display_single_prediction(predictions[0], metadata)
                    else:
                        # Affichage pour prédictions multiples
                        _display_multiple_predictions(new_data, predictions, metadata, selected_model_info)

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {str(e)}")
                    import traceback
                    with st.expander("Détails de l'erreur"):
                        st.code(traceback.format_exc())
    else:
        st.info("Remplissez le formulaire ou chargez un fichier pour générer des prédictions.")


def _render_prediction_form(metadata: dict) -> pd.DataFrame:
    st.markdown("### Remplissez les informations")

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
                            f"**{col_name}**",
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(mean_val),
                            help=f"Valeur entre {int(min_val)} et {int(max_val)}"
                        )
                    else:
                        form_values[col_name] = st.number_input(
                            f"**{col_name}**",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            format="%.2f",
                            help=f"Valeur entre {min_val:.2f} et {max_val:.2f}"
                        )

                elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                    unique_values = col_data.dropna().unique().tolist()
                    form_values[col_name] = st.selectbox(
                        f"**{col_name}**",
                        options=unique_values,
                        help=f"{len(unique_values)} valeurs possibles"
                    )

                elif pd.api.types.is_bool_dtype(col_data):
                    form_values[col_name] = st.checkbox(f"**{col_name}**", value=False)

                else:
                    form_values[col_name] = st.text_input(f"**{col_name}**", value="")

            else:
                form_values[col_name] = st.text_input(f"**{col_name}**", value="")

    with st.expander("Résumé des valeurs saisies"):
        summary_df = pd.DataFrame([form_values])
        st.dataframe(summary_df, use_container_width=True)

    return pd.DataFrame([form_values])


def _render_file_upload(selected_model_info: dict) -> pd.DataFrame:
    data_source = st.radio(
        "Source des données:",
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
                st.success(f"Dataset chargé: {new_data.shape[0]} lignes, {new_data.shape[1]} colonnes")

                with st.expander("Aperçu des données"):
                    st.dataframe(new_data.head(10))
            else:
                new_data = dataset.copy()
                st.success(f"Dataset chargé: {new_data.shape[0]} lignes, {new_data.shape[1]} colonnes")
        else:
            st.warning("Aucun dataset chargé.")

    else:  # Upload nouveau fichier
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

                st.success(f"Fichier chargé: {new_data.shape[0]} lignes, {new_data.shape[1]} colonnes")

                with st.expander("Aperçu des données"):
                    st.dataframe(new_data.head(10))

            except Exception as e:
                st.error(f"Erreur lors du chargement: {str(e)}")

    return new_data


def _display_single_prediction(prediction, metadata: dict):
    st.success("### Prédiction effectuée")

    target_column = metadata.get('target_column', 'Résultat')
    problem_type = metadata.get('problem_type', 'classification')

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if problem_type == 'classification':
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                <h3 style="margin: 0; color: white;">Prédiction pour</h3>
                <h2 style="margin: 10px 0; color: white;">{target_column}</h2>
                <h1 style="margin: 20px 0; font-size: 3em; color: white;">{prediction}</h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 15px; color: white;">
                <h3 style="margin: 0; color: white;">Prédiction pour</h3>
                <h2 style="margin: 10px 0; color: white;">{target_column}</h2>
                <h1 style="margin: 20px 0; font-size: 3em; color: white;">{prediction:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Informations sur le modèle utilisé")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Modèle", metadata.get('model_name', 'N/A'))
    with col2:
        st.metric("Type", problem_type.capitalize())
    with col3:
        metrics = metadata.get('metrics', {})
        if problem_type == 'classification':
            st.metric("Précision", f"{metrics.get('Accuracy', 0):.1%}")
        else:
            st.metric("R²", f"{metrics.get('R²', 0):.3f}")


def _display_multiple_predictions(new_data: pd.DataFrame, predictions, metadata: dict, selected_model_info: dict):
    result_df = new_data.copy()
    result_df[f'Prediction_{metadata["target_column"]}'] = predictions

    st.success(f"Prédictions générées avec succès pour {len(predictions)} lignes!")

    st.markdown("### Résultats des prédictions")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de prédictions", len(predictions))
    with col2:
        if metadata['problem_type'] == 'classification':
            unique_preds = np.unique(predictions)
            st.metric("Classes prédites", len(unique_preds))
        else:
            st.metric("Moyenne", f"{np.mean(predictions):.2f}")
    with col3:
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
            label="Télécharger en CSV",
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
            label="Télécharger en Excel",
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
