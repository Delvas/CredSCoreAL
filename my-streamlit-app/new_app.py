# Imports inutilisés supprimés
import streamlit as st
import numpy as np
import pickle
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title='CredScoreAL.IA',
    page_icon='CredScoreAl.ico',
    layout='centered',
    initial_sidebar_state='auto',
    menu_items={"Get Help": "mailto:fidelallou@gmail.com",
                "About": """
                    ## CredScoreAL.ia
                    Une plateforme innovante de credit scoring  💳  
                    Développée par **Fidel Allou**,**YOUSSOUF Y. TRAORE** ,**DARI O. MOHAMADOU**
                    📧 Contact : [fidelallou@gmail.com](mailto:fidelallou@gmail.com)  
                    🌐 Site web : [digital-pay.com](https://digital-pay.com)
                 """
                }
)

# Titre principal
st.title("🏦 CredScoreAL.ia : Prédiction de Score de Crédit")
st.markdown("---")

# accueil utilisateur
def show_user_name(utilisateur: str) -> None:
    st.info(f" **Bienvenu {utilisateur}**") 


utilisateur= st.sidebar.text_input(" Entrze votre nom svp")

# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_path: str) -> object:
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

# Fonction pour lister les modèles disponibles
@st.cache_data
def get_available_models(models_folder: str = "models") -> list:
    """Liste tous les modèles disponibles dans le dossier spécifié"""
    if not os.path.exists(models_folder):
        return []
    
    models = []
    for file in os.listdir(models_folder):
        if file.endswith(('.pkl', '.joblib')):
            models.append(file)
    
    return sorted(models)

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Configuration du dossier des modèles
    st.subheader("📁 Dossier des Modèles")
    models_folder = st.text_input(
        "Chemin du dossier des modèles:", 
        value="models",
        help="Dossier contenant les modèles (.pkl ou .joblib)"
    )
    
    # Chargement du modèle
    st.subheader("📂 Sélection du Modèle")
    
    # Rafraîchir la liste des modèles
    if st.button("🔄 Actualiser la liste"):
        st.cache_data.clear()
    
    # Liste des modèles disponibles
    available_models = get_available_models(models_folder)
    
    if available_models:
        selected_model = st.selectbox(
            "Choisissez un modèle:",
            options=["Aucun"] + available_models,
            help="Sélectionnez un modèle dans la liste"
        )
        if selected_model != "Aucun":
            model_path = os.path.join(models_folder, selected_model)
            model = load_model(model_path)
            if model is not None  :
                st.success("✅ Modèle chargé avec succès!")
                st.info(f"**Fichier:** {selected_model}")
                st.info(f"**Type:** {type(model).__name__}")
                
                # Afficher des informations supplémentaires si disponibles
                if hasattr(model, 'n_features_in_'):
                    st.info(f"**Variables d'entrée:** {model.n_features_in_}")
                if hasattr(model, 'classes_'):
                    st.info(f"**Classes:** {list(model.classes_)}")
            else:
                model = None
        else:
            model = None
    else:
        st.warning("❌ Aucun modèle trouvé")
        st.info(f"Vérifiez que le dossier '{models_folder}' existe et contient des fichiers .pkl ou .joblib")
        
        # Option de saisie manuelle si aucun modèle n'est trouvé
        st.subheader("📝 Saisie Manuelle")
        manual_path = st.text_input(
            "Chemin complet vers le modèle:", 
            placeholder="ex: /chemin/vers/model.pkl"
        )
        
        if manual_path and os.path.exists(manual_path):
            model = load_model(manual_path)
            if model:
                st.success("✅ Modèle chargé manuellement!")
                st.info(f"Type: {type(model).__name__}")
        else:
            model = None
            if manual_path:
                st.error("❌ Fichier non trouvé")

# Section d'information sur les modèles
if model is not None:
    with st.sidebar:
        st.markdown("---")
        st.subheader("ℹ️ Informations du Modèle")
        fichier_model = str(selected_model) if 'selected_model' in locals() and selected_model != "Aucun" else "Saisie manuelle"
        model_info = {"Type": str(type(model).__name__), "Fichier": fichier_model}
        # Ajout d'informations spécifiques avec conversion en str
        if hasattr(model, 'n_features_in_'):
            model_info["Variables d'entrée"] = str(getattr(model, 'n_features_in_', ''))
        if hasattr(model, 'classes_'):
            model_info["Classes"] = str(len(getattr(model, 'classes_', [])))
        if hasattr(model, 'feature_names_in_'):
            model_info["Variables nommées"] = "Oui"
        # Affichage des informations
        for key, value in model_info.items():
            st.text(f"{key}: {str(value)}")
        # Bouton pour voir les détails
        if st.button("📋 Détails du modèle"):
            st.json({
                "Type complet": str(type(model)),
                "Attributs disponibles": [attr for attr in dir(model) if not attr.startswith('_')],
                "Paramètres": str(getattr(model, 'get_params', lambda: "Non disponible")()) if hasattr(model, 'get_params') else "Non disponible"
            })

st.header("📋 Saisie des Données Client")
 # Formulaire de saisie
with st.form("my_form"):
    st.subheader("👤 Informations Personnelles")
    age = st.number_input("Âge", min_value=18, max_value=100, value=35)
    statut_propretaire = st.selectbox("Statut propriétaire", ["Propriétaire", "Locataire", "Autre"])
# Encodage simple : Propriétaire=2, Locataire=1, Autre=0
    statut_propretaire_enc = {"Propriétaire": 2, "Locataire": 1, "Autre": 0}[statut_propretaire]
    zone_habitat = st.number_input("Zone d'habitat : [Urbain:1, Rurale: 0]", min_value=0, max_value=1, value=0)
    Assurance_sante = st.number_input("Assurance Santé: [Oui:1, Non: 0]", min_value=0, max_value=1, value=0)
    if age < 25:
        Classe_age = 1
    elif age>=25 and age < 35 :
        Classe_age = 2
    elif age >=35 and age < 50 :
        Classe_age = 3
    else:
        Classe_age= 4

# Informations financières / bancaires
    st.subheader("[💰; 🏛️] Informations Financières / Bancaire")
    revenus_annuels = st.number_input("Revenus annuels", min_value=0,  value=45000, step=1000)
    type_pret = st.selectbox("Catégorie Prêt", ["Personnel", "Etude", "PME", "Immobilier", "Autre"])
    # Encodage simple : Personnel=0, Etude=1, PME=2, Immobilier=3, Autre=4
    type_pret_enc = {"Personnel": 0, "Etude": 1, "PME": 2, "Immobilier": 3, "Autre": 4}[type_pret]
    Historique_defaut = st.number_input(" Historique de defaut", min_value=0, value=1,step=1,max_value=1)
    duree_historique = st.number_input("Durée du prêt (années)", min_value=1, max_value=30, value=10)
    montant_demande = st.number_input("Montant demandé", min_value=1000, value=25000, step=1000)
    Taux_interet = st.number_input("Taux d'Intérêt (%)", min_value=0, max_value=100, step=1, value=30)

    #Taux_interet = st.number_input("Taux d'Intérêt", min_value=00, step=0.1 ,max_value=1,value=0.3)
    if revenus_annuels >= montant_demande*0.33 :
        cap_remb = 1
    else :
        cap_remb = 0 
    solde_moyen = st.number_input("Solde moyen ", min_value=0, value=5000, step=100 )
# Information alternatives
    recharge_mensuelle_moy=st.number_input("Recharge mensuelles moyenne", min_value=0,max_value=50,value=5)  
    solde_mobile_money_moy = st.number_input("Solde Mobile Money Moyen", min_value=0,max_value=2000000,value=1000)  
    whatsapp = st.number_input("Ancienneté Whatsapp(années)", min_value=0,max_value=50,value=5)
    Montant_facture_mensuel = st.number_input("Montant factures", min_value=0,max_value=5000000,value=5)
    ratio_facture_charge = 12 * (Montant_facture_mensuel / revenus_annuels  )
    indice_digit = ( whatsapp + recharge_mensuelle_moy + solde_mobile_money_moy) /3
    mature_finance = duree_historique / age

# Every form must have a submit button.
    submitted = st.form_submit_button(" Evaluer ")
    if submitted:
        if model is not None:
            # Si le modèle attend le taux d'intérêt entre 0 et 1, décommentez la ligne suivante
            # Taux_interet = Taux_interet / 100
            new = [
                age, statut_propretaire_enc, type_pret_enc, Historique_defaut,
                duree_historique, montant_demande, Taux_interet, cap_remb, solde_moyen,
                recharge_mensuelle_moy, solde_mobile_money_moy, whatsapp, Montant_facture_mensuel,
                ratio_facture_charge, indice_digit, mature_finance
            ]
            new = np.array(new).reshape(1, -1)
            prediction = model.predict(new)
            show_user_name(utilisateur)
            st.success(f"Votre score est de {prediction[0]}")
        else:
            st.error("Aucun modèle n'est chargé. Veuillez sélectionner ou charger un modèle.")