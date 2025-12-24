# ==========================================================
# CredScoreAL.IA - Application Streamlit de Credit Scoring
# Version Bâle II/III avec explicabilité + fallback
# Montants en CFA et seuil de risque configurable
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from PIL import Image
from datetime import datetime
from tensorflow import keras

from explain_credit import show_explanations   # Module externe pour raisons de refus

# --- Paramètres de l'application ---
icon = Image.open('CredScoreAl.ico')
st.set_page_config(
    page_title='CredScoreAL.IA',
    page_icon=icon,
    layout='centered',
    initial_sidebar_state='auto',
    menu_items={
        "Get Help": "mailto:fidelallou@gmail.com",
        "About": """
            ## CredScoreAL.ia
            Une plateforme innovante de credit scoring 💳  
            Développée par **Fidel Allou**, **YOUSSOUF Y. TRAORE**, **DARI O. MOHAMADOU**  
            📧 Contact : [fidelallou@gmail.com](mailto:fidelallou@gmail.com)  
            🌐 Site web : [digital-pay.com](https://digital-pay.com)
        """
    }
)

st.title("💳 Bienvenue sur CredScoreAL.ia")

# --- Sidebar utilisateur ---
utilisateur = st.sidebar.text_input("Entrez votre nom svp")
if utilisateur:
    st.sidebar.info(f"👤 Bonjour {utilisateur}")
st.sidebar.write("## 📅 Date et Heure Actuelles")
st.sidebar.write(f"**{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}**")

# --- Sidebar : Paramètres risque ---
st.sidebar.write("## ⚖️ Paramètres Risque (Bâle II/III)")
seuil_risque = st.sidebar.number_input(
    "Seuil de perte attendue (EL) en CFA",
    min_value=1_000,
    max_value=100_000_000,
    value=10_000_000,
    step=500_000
)

# --- Indicateur du mode explicatif ---
st.sidebar.markdown("## 🧾 Mode explicatif")
st.sidebar.success("Fallback activé ✅")

# --- Chargement des modèles ---
logit = joblib.load("models/LogReg_opt_08_2025.pkl")
RandomForest = joblib.load("models/Random Forest_best_model_optimise_08_2025.pkl")
Tree = joblib.load("models/DecisionTree_best_model_optimise_08_2025.pkl")
model_Dl = keras.models.load_model("models/Model_deepLearning_3.keras", compile=False)
encoder = joblib.load("models/encoder.pkl")

# --- Fonction utilitaire ---
def get_proba_accept(pred):  #n.array
    """Récupère correctement la proba d'acceptation selon la forme de sortie du modèle."""
    if len(pred.shape) > 1 and pred.shape[1] == 2:   # Modèle sklearn (2 colonnes)
        return pred[0][1]
    else:                         #[0,5]                  # Modèle DL avec une seule sortie
        return pred[0][0]

# --- Formulaire principal ---
with st.form("form_score"):
    st.subheader("👤 Informations Personnelles")
    age = st.number_input("Âge", min_value=18, max_value=100, value=35)
    statut_proprietaire = st.selectbox("Statut propriétaire", ["Propriétaire", "Locataire", "Autre"])
    statut_proprietaire_enc = {"Propriétaire": 2, "Locataire": 1, "Autre": 0}[statut_proprietaire]
    zone_habitat = st.selectbox("Zone d'habitat", ["Urbain", "Rurale"])
    zone_habitat = 1 if zone_habitat == "Urbain" else 0
    assurance_sante = st.radio("Assurance Santé", ["Oui", "Non"])
    assurance_sante = 1 if assurance_sante == "Oui" else 0

    if age < 25:
        Classe_age = 1
    elif age < 35:
        Classe_age = 2
    elif age < 50:
        Classe_age = 3
    else:
        Classe_age = 4

    st.subheader("💰 Informations Financières / Bancaires")
    revenus_annuels = st.number_input("Revenus annuels (CFA)", min_value=0, value=4_500_000, step=100_000)
    type_pret = st.selectbox("Catégorie Prêt", ["Personnel", "Etude", "PME", "Immobilier", "Autre"])
    type_pret_enc = {"Personnel": 0, "Etude": 1, "PME": 2, "Immobilier": 3, "Autre": 4}[type_pret]
    historique_defaut = st.number_input("Historique de défaut (0 ou 1)", min_value=0, max_value=1, value=0)
    montant_demande = st.number_input("Montant demandé (CFA)", min_value=100_000, value=2_500_000, step=100_000)
    duree_historique = st.number_input("Durée du prêt (années)", min_value=1, value=10)
    taux_interet = st.slider("Taux d'intérêt (%)", min_value=0, max_value=100, value=15) / 100

    cap_remb = 1 if revenus_annuels >= montant_demande * 0.33 else 0
    recharge_mensuelle_moy = st.number_input("Recharge mensuelle moyenne (CFA)", min_value=0, value=50_000, step=1000)
    solde_mobile_money_moy = st.number_input("Solde Mobile Money Moyen (CFA)", min_value=0, value=100_000, step=1000)
    whatsapp = st.number_input("Ancienneté Whatsapp (années)", min_value=0, value=5)
    facture_mensuelle = st.number_input("Montant factures mensuelles (CFA)", min_value=0, value=200_000, step=10000)
    ratio_facture_charge = 12 * (facture_mensuelle / revenus_annuels) if revenus_annuels > 0 else 0
    indice_digit = (whatsapp + recharge_mensuelle_moy + solde_mobile_money_moy) / 3
    mature_finance = duree_historique / age if age > 0 else 0

    # --- Données brutes pour fallback ---
    raw_input = [
        age, revenus_annuels, statut_proprietaire_enc, type_pret_enc,
        montant_demande, taux_interet, historique_defaut, duree_historique,
        recharge_mensuelle_moy, solde_mobile_money_moy, whatsapp, facture_mensuelle,
        zone_habitat, assurance_sante, Classe_age, cap_remb, ratio_facture_charge,
        indice_digit, mature_finance
    ]
    raw_df = pd.DataFrame([raw_input], columns=[
        "Age", "Revenu annuel", "Statut propriétaire", "Type prêt",
        "Montant demandé", "Taux intérêt", "Historique défaut", "Durée historique",
        "Recharge mensuelle", "Solde Mobile Money", "Whatsapp", "Factures mensuelles",
        "Zone habitat", "Assurance santé", "Classe âge", "Capacité remboursement",
        "Ratio factures/charges", "Indice digital", "Maturité financière"
    ])

    # --- Données normalisées pour les modèles ---
    new = np.array(raw_input).reshape(1, -1)
    cols_to_normalize = [1, 4, 5, 7, 8, 9, 10, 11, 16, 17, 18]
    try:
        new[:, cols_to_normalize] = encoder.transform(new[:, cols_to_normalize])
    except Exception:
        st.warning("⚠️ Normalisation ignorée (colonnes différentes du modèle).")

    submitted = st.form_submit_button("🔎 Soumettre", type="primary")

    if submitted:
        try:
            # Prédictions 
            predi_log = logit.predict_proba(new)      #[0,1]
            predi_randomF = RandomForest.predict_proba(new)
            predi_Tree = Tree.predict_proba(new)
            predi_dl = model_Dl.predict(new)  #[0]

            # Moyenne des probabilités
            proba_accept = (
                get_proba_accept(predi_log) +
                get_proba_accept(predi_randomF) +
                get_proba_accept(predi_Tree) +
                get_proba_accept(predi_dl)
            ) / 4

            st.write("### 📊 Calcul du Score de Crédit en cours...")
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)

            # --- Méthode Bâle II/III ---
            pd_client = 1 - proba_accept     # la proba d'être réfusé
            lgd = 0.45                       # perte en cas de défaut (45%)
            ead = montant_demande
            perte_attendue = pd_client * lgd * ead   #Lossi = Di × EADi × LGDi

            st.success(f"### Votre Score de crédit est {proba_accept*1000:.0f} ")
                #explique pourquoi pas éligible


            st.subheader("📊 Analyse Bâle II/III")
            st.write(f"- **Probabilité de défaut (PD)** : {pd_client:.2%}")
            st.write(f"- **Perte en cas de défaut (LGD)** : {lgd:.0%}")
            st.write(f"- **Exposition au défaut (EAD)** : {ead:,.0f} CFA")
            st.write(f"- **Perte attendue (EL)** : {perte_attendue:,.0f} CFA")
            st.write(f"- **Seuil de risque (configuré)** : {seuil_risque:,.0f} CFA")

            if perte_attendue < seuil_risque:
                st.success("✅ Crédit accordé : la perte attendue est inférieure au seuil de risque.")
                st.balloons()
            else:
                st.error("❌ Crédit refusé : la perte attendue dépasse le seuil de risque.")
                st.warning("👉 Raisons principales du refus :")
                show_explanations(RandomForest, raw_df, proba_accept, 0, st)

        except Exception as e:
            st.error(f"🚨 Une erreur est survenue : {e}")
            st.info("Vérifiez vos données ou contactez le support : fidelallou@gmail.com")

# --- Note réglementaire ---
st.markdown("---")
st.info("""
### 📑 Note réglementaire
L’évaluation du crédit repose sur les standards internationaux du **Comité de Bâle**  
(**Bâle II/III**), utilisés par les banques pour mesurer :

- **PD (Probability of Default)** : Probabilité de défaut  
- **LGD (Loss Given Default)** : Perte en cas de défaut  
- **EAD (Exposure At Default)** : Exposition au défaut  
- **EL (Expected Loss)** : Perte attendue  

Ces normes sont reconnues à l’échelle mondiale et servent de référence en matière de **gestion des risques bancaires**.
""")




