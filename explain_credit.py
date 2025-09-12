import pandas as pd

def fallback_explanations(X_input, st):
    """
    Explications simples du refus de crédit basées sur des règles métiers.
    Tous les montants sont affichés en FCFA.
    """
    st.subheader("📌 Raisons principales du refus (Fallback utilisé)")
    x = X_input.iloc[0]

    checks = []

    # Vérif revenu / montant demandé
    revenu = x["Revenu annuel"]
    montant = x["Montant demandé"]
    seuil_revenu = montant * 0.33
    if revenu < seuil_revenu:
        checks.append({
            "Critère": "Revenu annuel",
            "Valeur client": f"{revenu:,.0f} FCFA",
            "Seuil attendu": f"> {seuil_revenu:,.0f} FCFA",
            "Conclusion": "Revenu insuffisant"
        })

    # Ratio charges
    ratio = x["Ratio factures/charges"]
    if ratio > 0.4:
        checks.append({
            "Critère": "Ratio charges/revenus",
            "Valeur client": f"{ratio:.2f}",
            "Seuil attendu": "< 0.40",
            "Conclusion": "Charges trop élevées"
        })

    # Montant demandé
    seuil_montant = 5_000_000  # seuil en FCFA
    if montant > seuil_montant:
        checks.append({
            "Critère": "Montant demandé",
            "Valeur client": f"{montant:,.0f} FCFA",
            "Seuil attendu": f"< {seuil_montant:,.0f} FCFA",
            "Conclusion": "Montant trop élevé"
        })

    # Historique crédit
    duree = x["Durée historique"]
    if duree < 2:
        checks.append({
            "Critère": "Durée historique crédit",
            "Valeur client": f"{duree} ans",
            "Seuil attendu": "≥ 2 ans",
            "Conclusion": "Historique insuffisant"
        })

    # Cas par défaut
    if not checks:
        checks.append({
            "Critère": "Facteurs multiples",
            "Valeur client": "Profil global",
            "Seuil attendu": "-",
            "Conclusion": "Profil jugé risqué"
        })

    # Affichage sous forme de tableau
    df_checks = pd.DataFrame(checks)
    st.table(df_checks)


def show_explanations(model, X_input, proba, prediction, st):
    """
    Affiche dans Streamlit les raisons du refus ou de l’acceptation.
    Mentionne explicitement si le fallback est utilisé.
    """
    if prediction == 1:
        st.success(f"✅ Crédit accordé avec une probabilité de {proba:.2%}")
    else:
        st.error(f"❌ Crédit refusé (probabilité d’acceptation {proba:.2%})")
        st.info("ℹ️ Mode explicatif utilisé : **Fallback** (SHAP désactivé).")
        fallback_explanations(X_input, st)
