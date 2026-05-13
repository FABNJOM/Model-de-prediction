import streamlit as st
import requests

st.title("🛍️ Prédiction d'Achat")

st.text_area("Description :", "Prédisez si une personne va acheter selon son âge et son salaire")

st.subheader("Informations de la personne")

age = st.slider("Sélectionnez l'âge", 18, 100, 30)
salaire = st.slider("Sélectionnez le salaire (€)", 0, 200000, 50000, 5000)

nom = st.text_input("Entrez le nom de la personne: ")
st.info(f"👤 Client : {nom} (si renseigné)")

st.write(f"📊 Profil : {age} ans, {salaire:,} €")

if st.button("Prédire"):
    API_URL = "https://model-de-prediction-1.onrender.com/predict"
    
    with st.spinner("Analyse..."):
        try:
            response = requests.post(API_URL, json={"age": age, "salary": salaire})
            resultat = response.json()
            
            if resultat['achat']:
                st.success(f"✅ {resultat['message']} ! Probabilité : {resultat['confiance']['achat']}%")
            else:
                st.error(f"❌ {resultat['message']} ! Probabilité non-achat : {resultat['confiance']['non_achat']}%")
        except:
            st.error("Erreur de connexion à l'API")