#from calendar import c
from hmac import new
from click import confirm, option
from jwt import encode
from pyparsing import White
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from PIL import Image
from datetime import datetime
import time
from tensorflow import keras

# Importation de l'icône
icon = Image.open('CredScoreAl.ico')

# Paramètre de l'application
st.set_page_config(
    page_title='CredScoreAL.IA',
    page_icon=icon,
    layout='centered',
    initial_sidebar_state='auto',
    menu_items={"Get Help": "mailto:fidelallou@gmail.com",
                "About": """
                    ## CredScoreAL.ia
                    Une plateforme innovante de credit scoring  💳  
                    Développée par **Fidel Allou**,**YOUSSOUF Y. TRAORE** ,**DARI O. MOHAMADOU**
                    📧 Contact 1 : [fidelallou@gmail.com](mailto:fidelallou@gmail.com)
                    📧 Contact 2 : [yvtraore84@gmail.com](mailto:yvtraore84@.com)
                    📧 Contact 3 : [dari.mohamadou200@gmail.com](mailto:dari.mohamadou200@gmail.com)      
                    🌐 Site web : [digital-pay.com](https://digital-pay.com)
                 """
                }
)


# Titre de l'application Streamlit
st.title('Bienvenu sur CredScoreAL.ia ')

# Menu Sidebar
def show_user_name(utilisateur):
    st.info(f" **Hello {utilisateur}**")

st.sidebar.title('Menu Apli')
#clock =st.empty()
utilisateur = st.sidebar.text_input(" Entrez votre nom svp")
st.sidebar.write('---')

st.sidebar.write("## 📅 Date et Heure Actuelles")
now = datetime.now()
current_time = now.strftime("%d-%m-%y %H:%M:%S")
#st.slider.info(f" **{current_time}**")
st.sidebar.write(f"**{current_time}**")



# Chargement des modèles
logit = joblib.load("../models/LogisticRegression_best_model_optimise_08_2025.pkl")
RandomForest = joblib.load("../models/RandomForest_best_model_optimise_08_2025.pkl")
Tree = joblib.load("../models/DecisionTree_best_model_optimise_08_2025.pkl")
model_Dl = keras.models.load_model("../models/Model_deepLearning_3.keras")
encoder = joblib.load("../models/encoder.pkl")


# Formulaire principal toujours affiché
with st.form("form_score"):
    st.subheader("👤 Informations Personnelles")
    age = st.number_input("Âge", min_value=18, max_value=100, value=35)
    statut_propretaire = st.selectbox("Statut propriétaire", ["Propriétaire", "Locataire", "Autre"])
    statut_propretaire_enc = {"Propriétaire": 2, "Locataire": 1, "Autre": 0}[statut_propretaire]
    zone_habitat = st.number_input("Zone d'habitat : [Urbain:1, Rurale: 0]", min_value=0, max_value=1, value=0)
    Assurance_sante = st.number_input("Assurance Santé: [Oui:1, Non: 0]", min_value=0, max_value=1, value=0)
    if age < 25:
        Classe_age = 1
    elif age >= 25 and age < 35:
        Classe_age = 2
    elif age >= 35 and age < 50:
        Classe_age = 3
    else:
        Classe_age = 4

    st.subheader("[💰; 🏛️] Informations Financières / Bancaire")
    revenus_annuels = st.number_input("Revenus annuels", min_value=0, value=45000, step=1000)
    type_pret = st.selectbox("Catégorie Prêt", ["Personnel", "Etude", "PME", "Immobilier", "Autre"])
    type_pret_enc = {"Personnel": 0, "Etude": 1, "PME": 2, "Immobilier": 3, "Autre": 4}[type_pret]
    Historique_defaut = st.number_input("Historique de defaut", min_value=0, value=1, step=1, max_value=1)
    duree_historique = st.number_input("Durée du prêt (années)", min_value=0 ,value=10)#, max_value=30,
    montant_demande = st.number_input("Montant demandé", min_value=1000, value=25000, step=1000)
    Taux_interet = st.number_input("Taux d'Intérêt (%)", min_value=0, max_value=100, step=1, value=30)
    if revenus_annuels >= montant_demande * 0.33:
        cap_remb = 1
    else:
        cap_remb = 0
    #solde_moyen = st.number_input("Solde moyen", min_value=0, value=5000, step=100)
    recharge_mensuelle_moy = st.number_input("Recharge mensuelles moyenne", min_value=0, value=5)
    solde_mobile_money_moy = st.number_input("Solde Mobile Money Moyen", min_value=0, value=1000)
    whatsapp = st.number_input("Ancienneté Whatsapp(années)", min_value=0, value=5) #, max_value=50
    Montant_facture_mensuel = st.number_input("Montant factures", min_value=0, value=5) #, max_value=5000000
    ratio_facture_charge = 12 * (Montant_facture_mensuel / revenus_annuels)
    indice_digit = (whatsapp + recharge_mensuelle_moy + solde_mobile_money_moy) / 3
    mature_finance = duree_historique / age

    Taux_interet = Taux_interet / 100
    new = [
        age,revenus_annuels, statut_propretaire_enc, type_pret_enc,montant_demande,Taux_interet,
        Historique_defaut,duree_historique, recharge_mensuelle_moy, #solde_moyen,
        solde_mobile_money_moy, whatsapp, Montant_facture_mensuel,zone_habitat,Assurance_sante,
        Classe_age,cap_remb,ratio_facture_charge, indice_digit, mature_finance
    ]

    new = np.array(new).reshape(1, -1)
    # Normalisation
    cols_to_normalize = [1, 4, 5, 7, 8, 9, 10, 11, 16, 17, 18]  # adaptez selon l'ordre de vos features
    new[:, cols_to_normalize] = encoder.transform(new[:, cols_to_normalize])

    submitted = st.form_submit_button("Soumettre", type='primary')
    if submitted:
        try:
            predi_log     = logit.predict_proba(new)  
            predi_randomF = RandomForest.predict_proba(new)
            predi_Tree    = Tree.predict_proba(new)
            predi_dl      = model_Dl.predict(new)
            # Agrégation des prédictions (moyenne des probabilités)
            score_final = (predi_log*100 +  predi_randomF*100 + predi_Tree*100 + predi_dl*100) / 4
            score_arrondi = round(score_final[0][1]*10 , 0 )
            st.write("### Calcul du Score de Crédit en cours...")
            #simumation d'un calcul long
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            show_user_name(utilisateur)
            st.success(f"le score final est {score_arrondi} ")
            if score_arrondi >= 600  and score_arrondi < 700:
                st.balloons()
                st.info("Félicitation! Vous êtes éligible pour le crédit. Toutefois votre dossier mérite d'être examiner  🎉")
            elif score_arrondi >= 700 and score_arrondi <= 800:
                st.balloons()
                st.info("Félicitation! Vous êtes très éligible pour le crédit. 🎉")
            elif score_arrondi > 800 and score_arrondi <= 1000:
                st.balloons()
                st.info("Félicitation! Vous êtes hautement éligible pour le crédit. 🎉" )     
            else:

                st.warning("Désolé, vous n'êtes pas éligible pour le crédit. 😞")
                #explique pourquoi pas éligible
                
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction: {e}")
            st.info("Veuillez vérifier que tous les champs sont correctement remplis.")
            st.info("Si le problème persiste, contactez le support à l'adresse : fidelallou@gmail.com")

