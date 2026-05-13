from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="API de Prédiction d'Achat avec KNN")

# Autoriser les connexions externes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle et le scaler
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")

# Classe des données d'entrée
class Personne(BaseModel):
    age: float
    salary: float

# Route accueil
@app.get("/")
def accueil():
    return {
        "message": "Bienvenue sur l'API KNN 🚀"
    }

# Route prédiction
@app.post("/predict")
def predict(personne: Personne):

    # Données utilisateur
    data = np.array([[personne.age, personne.salary]])

    # Normalisation
    data_normalized = scaler.transform(data)

    # Prédiction
    prediction = model.predict(data_normalized)[0]

    # Probabilités
    probabilites = model.predict_proba(data_normalized)[0]

    confiance_non_achat = round(probabilites[0] * 100, 2)
    confiance_achat = round(probabilites[1] * 100, 2)

    return {
        "achat": bool(prediction),
        "message": (
            "Cette personne va acheter"
            if prediction == 1
            else "Cette personne ne va pas acheter"
        ),
        "confiance": {
            "achat": confiance_achat,
            "non_achat": confiance_non_achat
        }
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "OK"
    }