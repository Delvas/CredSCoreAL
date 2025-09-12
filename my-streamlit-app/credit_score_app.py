import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur de Score de Crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🏦 Système de Prédiction de Score de Crédit")
st.markdown("---")

# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_path):
    """Charge le modèle depuis le fichier spécifié"""
    try:
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
        elif model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            st.error("Format de fichier non supporté. Utilisez .pkl ou .joblib")
            return None
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Chargement du modèle
    st.subheader("📂 Modèle")
    model_path = st.text_input(
        "Chemin vers le modèle:", 
        placeholder="ex: models/credit_model.pkl",
        help="Chemin vers votre modèle pré-entraîné (.pkl ou .joblib)"
    )
    
    if model_path and os.path.exists(model_path):
        model = load_model(model_path)
        if model:
            st.success("✅ Modèle chargé avec succès!")
            st.info(f"Type: {type(model).__name__}")
    else:
        model = None
        if model_path:
            st.error("❌ Fichier non trouvé")

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Saisie des Données Client")
    
    # Formulaire de saisie
    with st.form("credit_form"):
        # Informations personnelles
        st.subheader("👤 Informations Personnelles")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=35)
            sexe = st.selectbox("Sexe", ["Masculin", "Féminin"])
            situation_familiale = st.selectbox(
                "Situation Familiale", 
                ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf(ve)"]
            )
        
        with col_p2:
            nb_enfants = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0)
            niveau_education = st.selectbox(
                "Niveau d'éducation",
                ["Primaire", "Secondaire", "Supérieur", "Post-graduée"]
            )
        
        # Informations financières
        st.subheader("💰 Informations Financières")
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            revenus_annuels = st.number_input(
                "Revenus annuels (€)", 
                min_value=0, 
                value=45000, 
                step=1000
            )
            type_emploi = st.selectbox(
                "Type d'emploi",
                ["CDI", "CDD", "Freelance", "Étudiant", "Retraité", "Chômeur"]
            )
            anciennete_emploi = st.number_input(
                "Ancienneté emploi (années)", 
                min_value=0, 
                max_value=50, 
                value=5
            )
        
        with col_f2:
            dettes_existantes = st.number_input(
                "Dettes existantes (€)", 
                min_value=0, 
                value=15000, 
                step=1000
            )
            montant_demande = st.number_input(
                "Montant demandé (€)", 
                min_value=1000, 
                value=25000, 
                step=1000
            )
            duree_pret = st.number_input(
                "Durée du prêt (années)", 
                min_value=1, 
                max_value=30, 
                value=10
            )
        
        # Informations bancaires
        st.subheader("🏛️ Historique Bancaire")
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            nb_comptes_bancaires = st.number_input(
                "Nombre de comptes bancaires", 
                min_value=1, 
                max_value=10, 
                value=2
            )
            solde_moyen = st.number_input(
                "Solde moyen (€)", 
                min_value=0, 
                value=5000, 
                step=100
            )
        
        with col_b2:
            nb_credits_precedents = st.number_input(
                "Nombre de crédits précédents", 
                min_value=0, 
                max_value=20, 
                value=1
            )
            retards_paiement = st.selectbox(
                "Historique de retards", 
                ["Aucun", "Occasionnels", "Fréquents"]
            )
        
        # Bouton de prédiction
        predict_button = st.form_submit_button("🔮 Prédire le Score de Crédit", type="primary")

# Colonnes des résultats
with col2:
    st.header("📊 Résultats")
    
    if predict_button and model is not None:
        # Préparation des données
        try:
            # Encodage des variables catégorielles (exemple)
            sexe_encoded = 1 if sexe == "Masculin" else 0
            
            # Mapping pour situation familiale
            situation_map = {"Célibataire": 0, "Marié(e)": 1, "Divorcé(e)": 2, "Veuf(ve)": 3}
            situation_encoded = situation_map[situation_familiale]
            
            # Mapping pour niveau d'éducation
            education_map = {"Primaire": 0, "Secondaire": 1, "Supérieur": 2, "Post-graduée": 3}
            education_encoded = education_map[niveau_education]
            
            # Mapping pour type d'emploi
            emploi_map = {"CDI": 0, "CDD": 1, "Freelance": 2, "Étudiant": 3, "Retraité": 4, "Chômeur": 5}
            emploi_encoded = emploi_map[type_emploi]
            
            # Mapping pour retards
            retard_map = {"Aucun": 0, "Occasionnels": 1, "Fréquents": 2}
            retard_encoded = retard_map[retards_paiement]
            
            # Calculs de ratios financiers
            ratio_dette_revenu = dettes_existantes / revenus_annuels if revenus_annuels > 0 else 0
            ratio_demande_revenu = montant_demande / revenus_annuels if revenus_annuels > 0 else 0
            
            # Création du vecteur de caractéristiques
            # Adaptez cette liste selon les variables de votre modèle
            features = np.array([[
                age, sexe_encoded, situation_encoded, nb_enfants, education_encoded,
                revenus_annuels, emploi_encoded, anciennete_emploi, dettes_existantes,
                montant_demande, duree_pret, nb_comptes_bancaires, solde_moyen,
                nb_credits_precedents, retard_encoded, ratio_dette_revenu, ratio_demande_revenu
            ]])
            
            # Prédiction
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features)[0]
                prediction = model.predict(features)[0]
                
                # Affichage des résultats avec probabilités
                st.success("✅ Prédiction réalisée!")
                
                # Score de crédit simulé (0-850)
                credit_score = int(300 + (prediction_proba.max() * 550))
                
                # Affichage du score
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = credit_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Score de Crédit"},
                    delta = {'reference': 650},
                    gauge = {
                        'axis': {'range': [None, 850]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [300, 580], 'color': "red"},
                            {'range': [580, 670], 'color': "orange"},
                            {'range': [670, 740], 'color': "yellow"},
                            {'range': [740, 850], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 650
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Interprétation du score
                if credit_score >= 740:
                    st.success("🟢 **Excellent** - Crédit approuvé avec conditions favorables")
                    risk_level = "Très faible"
                elif credit_score >= 670:
                    st.info("🟡 **Bon** - Crédit probablement approuvé")
                    risk_level = "Faible"
                elif credit_score >= 580:
                    st.warning("🟠 **Moyen** - Crédit à examiner")
                    risk_level = "Modéré"
                else:
                    st.error("🔴 **Faible** - Risque élevé de refus")
                    risk_level = "Élevé"
                
                # Détails supplémentaires
                st.write(f"**Niveau de risque:** {risk_level}")
                if hasattr(model, 'predict_proba') and len(prediction_proba) == 2:
                    st.write(f"**Probabilité d'approbation:** {prediction_proba[1]:.1%}")
                    st.write(f"**Probabilité de refus:** {prediction_proba[0]:.1%}")
            
            else:
                # Modèle sans probabilités
                prediction = model.predict(features)[0]
                st.success("✅ Prédiction réalisée!")
                
                if prediction == 1:
                    st.success("🟢 **Crédit Approuvé**")
                else:
                    st.error("🔴 **Crédit Refusé**")
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
            st.info("Vérifiez que les variables correspondent à celles utilisées pour entraîner votre modèle.")
    
    elif predict_button and model is None:
        st.error("❌ Veuillez d'abord charger un modèle valide.")

# Section d'analyse des facteurs
if model is not None:
    st.markdown("---")
    st.header("📈 Analyse des Facteurs de Risque")
    
    # Graphiques informatifs
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Graphique ratio dette/revenu
        fig_ratio = px.bar(
            x=['Ratio Dette/Revenu', 'Ratio Demande/Revenu'],
            y=[ratio_dette_revenu if 'ratio_dette_revenu' in locals() else 0, 
               ratio_demande_revenu if 'ratio_demande_revenu' in locals() else 0],
            title="Ratios Financiers",
            color=['Ratio Dette/Revenu', 'Ratio Demande/Revenu']
        )
        fig_ratio.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_ratio, use_container_width=True)
    
    with col_g2:
        # Répartition des revenus
        if 'revenus_annuels' in locals():
            revenus_net = revenus_annuels - dettes_existantes
            fig_pie = px.pie(
                values=[dettes_existantes, revenus_net],
                names=['Dettes Existantes', 'Revenus Net'],
                title="Répartition Financière"
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; font-size: 0.8em;'>
    🏦 Système de Prédiction de Score de Crédit | 
    Développé avec Streamlit | 
    ⚠️ Outil d'aide à la décision uniquement
</div>
""", unsafe_allow_html=True)

# Instructions d'utilisation
with st.expander("📖 Instructions d'utilisation"):
    st.markdown("""
    ### Comment utiliser cette application:
    
    1. **Chargement du modèle:** 
       - Entrez le chemin vers votre modèle pré-entraîné dans la barre latérale
       - Formats supportés: .pkl (pickle) et .joblib
    
    2. **Saisie des données:**
       - Remplissez tous les champs du formulaire
       - Les données sont organisées par catégories pour faciliter la saisie
    
    3. **Prédiction:**
       - Cliquez sur "Prédire le Score de Crédit"
       - Le système affichera le score et les probabilités
    
    ### Variables utilisées:
    - **Personnelles:** âge, sexe, situation familiale, nombre d'enfants, éducation
    - **Financières:** revenus, type d'emploi, ancienneté, dettes, montant demandé
    - **Bancaires:** comptes, solde moyen, historique de crédit
    
    ### Notes importantes:
    - Adaptez les variables selon votre modèle spécifique
    - Vérifiez que l'encodage des variables correspond à votre entraînement
    - Cet outil est une aide à la décision, pas un système de décision automatique
    """)
