from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")

class Personne(BaseModel):
    age: float
    salary: float

@app.post("/predict")
def predict(personne: Personne):
    data = np.array([[personne.age, personne.salary]])
    data_norm = scaler.transform(data)
    prediction = model.predict(data_norm)[0]
    proba = model.predict_proba(data_norm)[0]
    
    return {
        "achat": bool(prediction),
        "message": "Achètera" if prediction == 1 else "N'achètera pas",
        "confiance": {
            "achat": round(proba[1] * 100, 2),
            "non_achat": round(proba[0] * 100, 2)
        }
    }

@app.get("/")
def root():
    return {"status": "API OK", "endpoint": "/predict"}