"""Page Chat — Conversation avec les données via LLM + RAG"""
import streamlit as st
from backend.utils.llm_client import call_llm, is_available
from backend.utils.rag_engine import get_rag
from backend.utils.rag_evaluator import RAGEvaluator
from frontend.utils.ui_helpers import page_header


def render_chat():
    page_header("💬", "Chat avec vos données",
                "Posez des questions en langage naturel sur votre dataset",
                badge="IA + RAG")

    # ─── Vérifications ────────────────────────────────────────────────────────
    if not is_available():
        st.markdown(
            '<div style="background:#fef2f2;border-radius:14px;padding:28px;'
            'border:1px solid #fecaca;border-left:4px solid #ef4444;'
            'text-align:center;margin:20px 0;">'
            '<div style="font-size:40px;margin-bottom:12px;">🔑</div>'
            '<div style="font-size:15px;font-weight:700;color:#dc2626;margin-bottom:8px;">'
            'Clé API manquante</div>'
            '<div style="font-size:13px;color:#991b1b;margin-bottom:16px;">'
            'Configurez votre clé Anthropic dans la sidebar pour activer le chat.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        return

    if not st.session_state.get('dataset_loaded', False):
        st.markdown(
            '<div style="background:#fffbeb;border-radius:14px;padding:28px;'
            'border:1px solid #fde68a;border-left:4px solid #f59e0b;'
            'text-align:center;margin:20px 0;">'
            '<div style="font-size:40px;margin-bottom:12px;">📁</div>'
            '<div style="font-size:15px;font-weight:700;color:#92400e;margin-bottom:8px;">'
            'Aucune donnée chargée</div>'
            '<div style="font-size:13px;color:#78350f;">'
            'Chargez d\'abord un dataset pour pouvoir interroger vos données.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("← Charger un dataset", type="primary"):
            st.session_state.current_page = "home"
            st.rerun()
        return

    # ─── Initialisation de l'historique ───────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ─── Indexation RAG (une seule fois par dataset) ───────────────────────────
    info = st.session_state.dataset_info
    current_filename = info.get('filename', '') if info else ''
    rag = get_rag()

    df = st.session_state.shared_dataset
    if rag.needs_reindex(df, current_filename):
        st.session_state.pop('chat_dataset_context', None)
        with st.spinner("Indexation RAG du dataset en cours..."):
            rag.index(df, filename=current_filename)

    # ─── Affichage de l'en-tête dataset ───────────────────────────────────────
    method_badge = rag.method()
    st.markdown(
        f'<div style="background:#eff6ff;border-radius:10px;padding:12px 18px;'
        f'border:1px solid #bfdbfe;margin-bottom:20px;display:flex;align-items:center;'
        f'justify-content:space-between;gap:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:16px;">📊</span>'
        f'<span style="font-size:13px;color:#1d4ed8;">'
        f'Contexte actif : <strong>{info.get("filename", "dataset")}</strong>'
        f' — {info.get("rows", 0):,} lignes × {info.get("columns", 0)} colonnes'
        f'</span></div>'
        f'<span style="background:#1d4ed815;color:#1d4ed8;border:1px solid #1d4ed830;'
        f'border-radius:12px;padding:2px 10px;font-size:11px;font-weight:600;">'
        f'RAG / {method_badge}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ─── Onglets Chat / Évaluation RAG ───────────────────────────────────────
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Évaluation RAG"])

    with tab_chat:
        _render_chat_tab(rag, info)

    with tab_eval:
        _render_eval_tab(rag)


def _render_chat_tab(rag, info):
    """Contenu de l'onglet Chat."""
    if not st.session_state.chat_history:
        st.markdown(
            '<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:12px;">'
            'Exemples de questions</div>',
            unsafe_allow_html=True
        )
        examples = [
            "Quelles colonnes ont le plus de valeurs manquantes ?",
            "Quel algorithme ML recommandes-tu pour ce dataset ?",
            "Y a-t-il des corrélations importantes entre les variables ?",
            "Montre-moi un résumé des colonnes numériques.",
        ]
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, use_container_width=True, key=f"ex_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": example})
                    _generate_response(example)
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ex : Quelle est la variable la plus corrélée à la cible ?")
    if user_input and user_input.strip():
        question = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": question})
        _generate_response(question)
        st.rerun()

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Effacer la conversation", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


def _render_eval_tab(rag):
    """Onglet d'évaluation du RAG — sans appel LLM."""
    evaluator = RAGEvaluator()
    df = st.session_state.shared_dataset

    # ── Section 1 : test d'une requête libre ──────────────────────────────────
    st.markdown(
        '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px;">'
        'Test de retrieval</div>'
        '<div style="font-size:13px;color:#64748b;margin-bottom:16px;">'
        'Saisissez une question pour voir quels chunks sont retournés et leur score.</div>',
        unsafe_allow_html=True
    )

    test_query = st.text_input(
        "Question de test",
        placeholder="Ex : Quelle colonne a le plus de valeurs manquantes ?",
        label_visibility="collapsed",
        key="eval_query"
    )

    k_val = st.slider("Nombre de chunks (k)", min_value=1, max_value=10, value=6, key="eval_k")

    if test_query and test_query.strip():
        result = evaluator.evaluate_query(rag, test_query.strip(), k=k_val)

        # Faithfulness + Answer Relevance depuis la dernière réponse du chat
        last_answer = next(
            (m for m in reversed(st.session_state.get("chat_history", []))
             if m["role"] == "assistant" and m.get("_question") == test_query.strip()),
            None
        )
        if last_answer:
            faith = evaluator.evaluate_faithfulness(last_answer["content"], result.chunks)
            ar    = evaluator.evaluate_answer_relevance(test_query.strip(), last_answer["content"])
        else:
            faith, ar = None, None

        # Métriques retrieval
        verdict_color = {
            "Excellent": "#22c55e", "Bon": "#f59e0b",
            "Moyen": "#f97316", "Faible": "#ef4444"
        }.get(result.verdict, "#94a3b8")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Score retrieval", f"{result.retrieval_score:.2%}",
                      help="Similarité cosinus moyenne des top-k chunks (0→1)")
        with c2:
            st.metric("Precision@k", f"{result.precision_at_k:.2%}",
                      help="% chunks contenant ≥1 mot-clé (mot entier)")
        with c3:
            st.metric("MRR", f"{result.mrr:.2%}",
                      help="Mean Reciprocal Rank : 1/rang du 1er chunk pertinent")
        with c4:
            st.metric("Recall (proxy)", f"{result.keyword_coverage:.2%}",
                      help="% mots de la question trouvés dans les chunks (mot entier)")
        with c5:
            st.metric("Discrimination", f"{result.score_gap:.2%}",
                      help="Écart max-min des scores — élevé = retrieval ciblé")

        # Métriques post-génération (si réponse disponible)
        if faith is not None:
            st.markdown(
                '<div style="font-size:12px;color:#64748b;margin:8px 0 4px 0;font-weight:600;">'
                'Métriques post-génération (basées sur la dernière réponse du chat)</div>',
                unsafe_allow_html=True
            )
            cfa, car = st.columns(2)
            faith_color = "#22c55e" if faith >= 0.6 else "#f59e0b" if faith >= 0.4 else "#ef4444"
            ar_color   = "#22c55e" if ar   >= 0.5 else "#f59e0b" if ar   >= 0.3 else "#ef4444"
            with cfa:
                st.metric("Faithfulness", f"{faith:.2%}",
                          help="% mots de la réponse ancrés dans le contexte RAG (proxy anti-hallucination)")
            with car:
                st.metric("Answer Relevance", f"{ar:.2%}",
                          help="Chevauchement vocabulaire question ↔ réponse")

        st.markdown(
            f'<div style="background:{verdict_color}15;border:1px solid {verdict_color}40;'
            f'border-radius:8px;padding:8px 14px;margin:8px 0;font-size:13px;font-weight:600;color:{verdict_color};">'
            f'Verdict : {result.verdict}</div>',
            unsafe_allow_html=True
        )

        # Détail des chunks
        with st.expander("Voir les chunks retournés"):
            for i, (chunk, score) in enumerate(zip(result.chunks, result.scores)):
                score_color = "#22c55e" if score >= 0.5 else "#f59e0b" if score >= 0.3 else "#ef4444"
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0;border-left:3px solid {score_color};'
                    f'border-radius:6px;padding:10px 14px;margin-bottom:8px;">'
                    f'<span style="font-size:11px;color:{score_color};font-weight:700;">'
                    f'Chunk {i+1} — score {score:.4f}</span></div>',
                    unsafe_allow_html=True
                )
                st.code(chunk[:400] + ("..." if len(chunk) > 400 else ""), language=None)

    st.divider()

    # ── Section 2 : benchmark auto-généré ────────────────────────────────────
    st.markdown(
        '<div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px;">'
        'Benchmark automatique</div>'
        '<div style="font-size:13px;color:#64748b;margin-bottom=16px;">'
        'Questions auto-générées depuis le dataset pour mesurer la qualité globale du RAG.</div>',
        unsafe_allow_html=True
    )

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        run_benchmark = st.button("Lancer le benchmark", type="primary", use_container_width=True)

    if run_benchmark or st.session_state.get("_benchmark_report"):
        if run_benchmark:
            with st.spinner("Benchmark en cours..."):
                report = evaluator.run_benchmark(rag, df, k=6)
            st.session_state["_benchmark_report"] = report
        else:
            report = st.session_state["_benchmark_report"]

        # Scores globaux
        overall_color = "#22c55e" if report.overall_score >= 0.55 else \
                        "#f59e0b" if report.overall_score >= 0.35 else "#ef4444"
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Score global", f"{report.overall_score:.2%}",
                      help="Similarité cosinus moyenne toutes questions")
        with c2:
            st.metric("Precision@k", f"{report.mean_precision:.2%}",
                      help="% moyen de chunks pertinents parmi les k retournés")
        with c3:
            st.metric("MRR", f"{report.mean_mrr:.2%}",
                      help="Mean Reciprocal Rank moyen — mesure la qualité du classement")
        with c4:
            st.metric("Success Rate", f"{report.mean_success_rate:.2%}",
                      help="% de questions avec au moins 1 chunk pertinent trouvé")
        with c5:
            st.metric("Questions testées", len(report.questions))

        # Scores par catégorie
        st.markdown(
            '<div style="font-size:13px;font-weight:600;color:#374151;margin:16px 0 8px 0;">'
            'Scores par catégorie</div>',
            unsafe_allow_html=True
        )
        cat_labels = {"schema": "Schéma", "stats": "Statistiques",
                      "qualite": "Qualité", "correlations": "Corrélations"}
        cat_cols = st.columns(len(report.per_category))
        for col, (cat, score) in zip(cat_cols, report.per_category.items()):
            c = "#22c55e" if score >= 0.55 else "#f59e0b" if score >= 0.35 else "#ef4444"
            with col:
                st.metric(cat_labels.get(cat, cat), f"{score:.2%}")

        # Détail par question
        with st.expander("Détail par question"):
            for q, r in zip(report.questions, report.results):
                verdict_color = {
                    "Excellent": "#22c55e", "Bon": "#f59e0b",
                    "Moyen": "#f97316", "Faible": "#ef4444"
                }.get(r.verdict, "#94a3b8")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:8px 12px;border-bottom:1px solid #f1f5f9;">'
                    f'<span style="font-size:12px;color:#374151;flex:1;">{q.query}</span>'
                    f'<div style="display:flex;gap:14px;align-items:center;flex-shrink:0;">'
                    f'<span style="font-size:11px;color:#94a3b8;" title="Retrieval">↕ {r.retrieval_score:.2%}</span>'
                    f'<span style="font-size:11px;color:#94a3b8;" title="Precision@k">P@k {r.precision_at_k:.2%}</span>'
                    f'<span style="font-size:11px;color:#94a3b8;" title="MRR">MRR {r.mrr:.2%}</span>'
                    f'<span style="background:{verdict_color}15;color:{verdict_color};'
                    f'border-radius:10px;padding:2px 8px;font-size:11px;font-weight:600;">'
                    f'{r.verdict}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        # Config utilisée
        st.markdown(
            f'<div style="background:#f8fafc;border-radius:8px;padding:10px 14px;'
            f'margin-top:12px;font-size:12px;color:#64748b;">'
            f'Configuration : méthode={report.config["method"]} | '
            f'modèle={report.config["model"]} | '
            f'k={report.config["k"]} | '
            f'chunks indexés={report.config["n_chunks"]}</div>',
            unsafe_allow_html=True
        )

        # Export CSV
        import pandas as _pd
        rows = []
        for q, r in zip(report.questions, report.results):
            rows.append({
                "question":        q.query,
                "categorie":       q.category,
                "retrieval_score": r.retrieval_score,
                "precision_at_k":  r.precision_at_k,
                "mrr":             r.mrr,
                "recall_proxy":    r.keyword_coverage,
                "success_rate":    r.success_rate,
                "verdict":         r.verdict,
            })
        csv_data = _pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger le rapport benchmark (CSV)",
            data=csv_data,
            file_name="rag_benchmark.csv",
            mime="text/csv",
            use_container_width=False,
        )


def _generate_response(question: str):
    """Appelle Claude avec contexte RAG et ajoute la réponse à l'historique."""
    rag = get_rag()
    df  = st.session_state.shared_dataset

    # Contexte dynamique : chunks pertinents pour CETTE question
    system = rag.build_context(question, df)

    # Historique multi-tour — limité aux 10 derniers échanges pour éviter le dépassement de contexte
    MAX_HISTORY_TURNS = 10
    history = st.session_state.chat_history[-(MAX_HISTORY_TURNS * 2):]
    conversation = ""
    for msg in history[:-1]:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n\n"

    full_prompt = conversation + f"Utilisateur: {question}"

    with st.spinner("Analyse en cours..."):
        response = call_llm(
            prompt=full_prompt,
            system=system,
        )

    # Stocker la réponse + les chunks utilisés pour Faithfulness/Answer Relevance
    context_chunks = rag.search(question, k=6)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response if response else
            "Je n'ai pas pu générer une réponse. Vérifiez votre connexion et votre clé API.",
        "_chunks": context_chunks,
        "_question": question,
    })
