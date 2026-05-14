import streamlit as st
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="Prédiction d'Achat",
    page_icon="🛍️",
    layout="centered"
)

# Titre principal
st.title("🛍️ Prédiction d'Achat")
st.markdown("### Entrez l'âge et le salaire pour savoir si la personne va acheter")
st.markdown("---")

# URL de ton API déployée sur Render
API_URL = "https://api-achat.onrender.com/predict"

# Création des deux colonnes pour les inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "📅 Âge", 
        min_value=18, 
        max_value=100, 
        value=30, 
        step=1,
        help="Âge de la personne entre 18 et 100 ans"
    )

with col2:
    salaire = st.number_input(
        "💰 Salaire estimé (€)", 
        min_value=0, 
        max_value=200000, 
        value=50000, 
        step=5000,
        help="Salaire annuel en euros"
    )

# Afficher le profil
st.info(f"📌 Profil sélectionné : {age} ans, {salaire:,} €")

# Bouton de prédiction
if st.button("🔮 Prédire l'achat", type="primary", use_container_width=True):
    with st.spinner("Analyse en cours..."):
        try:
            # Appel à l'API
            response = requests.post(
                API_URL, 
                json={"age": age, "salary": salaire},
                timeout=60
            )
            
            if response.status_code == 200:
                resultat = response.json()
                
                st.markdown("---")
                st.subheader("📊 Résultat de la prédiction")
                
                # Afficher le résultat avec des couleurs
                if resultat['achat']:
                    st.success(f"### ✅ {resultat['message']}")
                    st.balloons()  # Petite animation de succès
                else:
                    st.error(f"### ❌ {resultat['message']}")
                
                # Barres de progression
                col_a, col_b = st.columns(2)
                with col_a:
                    confiance_achat = resultat['confiance']['achat']
                    st.metric("📈 Probabilité d'achat", f"{confiance_achat}%")
                    st.progress(confiance_achat / 100)
                
                with col_b:
                    confiance_non_achat = resultat['confiance']['non_achat']
                    st.metric("📉 Probabilité de non-achat", f"{confiance_non_achat}%")
                    st.progress(confiance_non_achat / 100)
                
            else:
                st.error(f"❌ Erreur API : {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.warning("⏰ Le service se réveille... Réessaie dans quelques secondes.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Impossible de se connecter à l'API. Vérifie que l'API est bien déployée.")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# Pied de page
st.markdown("---")
st.caption(f"🔗 API utilisée : `{API_URL}`")
st.caption("🏆 Modèle KNN entraîné sur le dataset Social Network Ads")