from fastapi import FastAPI
import joblib
import os
import pandas as pd

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
modelo = joblib.load(os.path.join(BASE_DIR, "modelo_asistencias.pkl"))

@app.get("/")
def home():
    return {"status": "API funcionando"}

@app.post("/predict")
def predict(data: dict):

    alumno_id = data["alumno_id"]
    estado = data["estado"]

    nuevo = pd.DataFrame([{
        "alumno_id": alumno_id,
        "estado_num": estado
    }])

    pred = modelo.predict(nuevo)[0]
    prob = modelo.predict_proba(nuevo)[0]

    return {
        "prediccion": int(pred),
        "presente": float(prob[0]),
        "tarde": float(prob[1]),
        "falta": float(prob[2])
    }
