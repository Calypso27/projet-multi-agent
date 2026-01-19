"""Page A propos"""
import streamlit as st


def render_about():
    """Page a propos du systeme"""
    
    st.markdown("# A Propos")
    
    st.markdown("""
    ## Plateforme d'Analyse de Donnees Multi-Agent
    
    Cette plateforme utilise une architecture multi-agent pour rendre l'analyse de donnees
    et la creation de modeles predictifs accessibles a tous, meme sans expertise technique.
    
    ### Architecture
    
    Le systeme est compose de 4 agents intelligents qui collaborent:
    
    **1. Chef de Projet**
    - Coordonne l'ensemble du systeme
    - Traduit vos demandes en langage naturel
    - Synthetise les resultats
    
    **2. Ingenieur Donnees**
    - Charge vos fichiers (CSV, Excel, JSON, etc.)
    - Valide la qualite des donnees
    - Detecte automatiquement les problemes
    
    **3. Analyste**
    - Effectue les analyses statistiques
    - Detecte les correlations
    - Genere des insights
    
    **4. Modelisateur ML**
    - Entraine automatiquement des modeles
    - Compare plusieurs algorithmes
    - Selectionne le meilleur modele
    
    ### Fonctionnalites
    
    - **Support multi-format**: CSV, Excel, JSON, Parquet, TSV
    - **Analyse automatique**: Statistiques, correlations, qualite des donnees
    - **Prediction guidee**: Assistant pas-a-pas pour creer des modeles
    - **ML automatise**: Entrainement et comparaison de modeles
    - **Interface intuitive**: Pas besoin d'etre expert en data science
    
    ### Technologies
    
    - **Frontend**: Streamlit
    - **Backend**: Python, Pandas, Scikit-learn
    - **Architecture**: Systeme multi-agent avec bus de messages
    - **ML**: Regression et Classification automatisees
    
    ### Version
    
    Version 2.0 - Semaine 3
    
    Developpe dans le cadre du projet de Plateforme Conversationnelle d'Analyse de Donnees
    """)
    
    st.markdown("---")
    
    st.markdown("### Statut du Systeme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Agents Actifs:**")
        st.markdown("- Chef de Projet OK")
        st.markdown("- Ingenieur Donnees OK")
        st.markdown("- Analyste OK")
        st.markdown("- Modelisateur ML OK")
    
    with col2:
        st.markdown("**Fonctionnalites:**")
        st.markdown("- Upload multi-format OK")
        st.markdown("- Analyse exploratoire OK")
        st.markdown("- Entrainement ML OK")
        st.markdown("- Interface guidee OK")
