from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI(title="API de Prédiction d'Achat avec KNN")

# Charger le modèle et le scaler (pas d'encoder car c'est déjà 0/1)
model = joblib.load("model_knn.pkl")  # Ton modèle KNN entraîné
scaler = joblib.load("scaler.pkl")     # Le StandardScaler que tu as utilisé

# Définir la classe pour les données d'entrée
class Personne(BaseModel):
    age: float          # Âge de la personne
    salary: float       # EstimatedSalary (salaire estimé)

@app.get("/")
def accueil():
    return {
        "message": "Bienvenue sur l'API de Prédiction d'Achat avec KNN!",
        "info": "Envoyez l'âge et le salaire pour savoir si la personne va acheter"
    }

@app.post("/predict")
def predict(personne: Personne):
    # Préparer les données
    data = np.array([[personne.age, personne.salary]])
    
    # Normaliser les données (important !)
    data_normalized = scaler.transform(data)
    
    # Prédire
    prediction = model.predict(data_normalized)[0]
    
    # Résultat
    if prediction == 1:
        resultat = "Achète"
        message_detail = "✅ Cette personne va ACHETER le produit"
    else:
        resultat = "N'achète pas"
        message_detail = "❌ Cette personne ne va PAS acheter le produit"
    
    return {
        "age": personne.age,
        "salary": personne.salary,
        "prediction": int(prediction),
        "resultat": resultat,
        "message": message_detail,
        "probabilites": model.predict_proba(data_normalized)[0].tolist()
    }

@app.post("/predict-batch")
def predict_batch(personnes: list[Personne]):
    """Prédire pour plusieurs personnes à la fois"""
    results = []
    
    for personne in personnes:
        data = np.array([[personne.age, personne.salary]])
        data_normalized = scaler.transform(data)
        prediction = model.predict(data_normalized)[0]
        
        results.append({
            "age": personne.age,
            "salary": personne.salary,
            "prediction": int(prediction),
            "achat": "Oui" if prediction == 1 else "Non"
        })
    
    return {"predictions": results}

@app.get("/health")
def health_check():
    return {
        "status": "API en bonne santé",
        "model_charge": model is not None,
        "scaler_charge": scaler is not None
    }

# Pour lancer l'API : uvicorn main:app --reload