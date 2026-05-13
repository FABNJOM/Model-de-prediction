import streamlit as st
import numpy as np
import joblib

# Charger les modèles
@st.cache_resource
def load_models():
    model = joblib.load("model_knn.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

# Titre
st.title("🛍️ Prédiction d'Achat")
st.write("Entrez l'âge et le salaire pour savoir si la personne va acheter")

# Entrées utilisateur
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge:", min_value=18, max_value=100, value=30)

with col2:
    salaire = st.number_input("Salaire (€):", min_value=0, max_value=200000, value=50000)

# Bouton de prédiction
if st.button("Prédire"):
    # Préparer et normaliser
    personne = np.array([[age, salaire]])
    personne_norm = scaler.transform(personne)
    
    # Prédire
    prediction = model.predict(personne_norm)[0]
    probabilite = model.predict_proba(personne_norm)[0][1]
    
    # Afficher résultat
    st.subheader("Résultat :")
    if prediction == 1:
        st.success(f"✅ Cette personne va ACHETER le produit (probabilité: {probabilite*100:.1f}%)")
    else:
        st.error(f"❌ Cette personne ne va PAS acheter le produit (probabilité de non-achat: {(1-probabilite)*100:.1f}%)")
    
    # Barre de progression
    st.progress(probabilite, text=f"Probabilité d'achat: {probabilite*100:.1f}%")