from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(
    title="API de Prédiction d'Achat",
    description="Prédit si une personne va acheter un produit selon son âge et son salaire",
    version="1.0"
)

# Charger le modèle et le scaler
try:
    model = joblib.load("model_knn.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Modèles chargés avec succès")
except Exception as e:
    print(f"❌ Erreur de chargement: {e}")
    model = None
    scaler = None

class Personne(BaseModel):
    age: float
    salary: float

@app.get("/")
def accueil():
    return {
        "message": "API de Prédiction d'Achat avec KNN",
        "endpoints": {
            "POST /predict": "Prédire pour une personne",
            "POST /predict-batch": "Prédire pour plusieurs personnes",
            "GET /health": "Vérifier l'état",
            "GET /docs": "Documentation interactive"
        }
    }

@app.post("/predict")
def predict(personne: Personne):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    # Préparer et normaliser
    data = np.array([[personne.age, personne.salary]])
    data_normalized = scaler.transform(data)
    
    # Prédire
    prediction = model.predict(data_normalized)[0]
    probabilities = model.predict_proba(data_normalized)[0]
    
    return {
        "age": personne.age,
        "salary": personne.salary,
        "achat": bool(prediction),
        "message": "Achètera" if prediction == 1 else "N'achètera pas",
        "confiance": {
            "non_achat": round(probabilities[0] * 100, 2),
            "achat": round(probabilities[1] * 100, 2)
        }
    }

@app.post("/predict-batch")
def predict_batch(personnes: List[Personne]):
    results = []
    for personne in personnes:
        data = np.array([[personne.age, personne.salary]])
        data_normalized = scaler.transform(data)
        prediction = model.predict(data_normalized)[0]
        
        results.append({
            "age": personne.age,
            "salary": personne.salary,
            "achat": "Oui" if prediction == 1 else "Non"
        })
    
    return {"total": len(results), "predictions": results}

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }