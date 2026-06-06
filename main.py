from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# cargar modelo
BASE_DIR = os.path.dirname(__file__)
modelo = joblib.load(os.path.join(BASE_DIR, "modelo_asistencias.pkl"))

class InputData(BaseModel):
    alumno_id: int
    estado: int

@app.post("/predict")
def predict(data: InputData):

    nuevo = pd.DataFrame([{
        "alumno_id": data.alumno_id,
        "estado_num": data.estado
    }])

    pred = modelo.predict(nuevo)[0]
    prob = modelo.predict_proba(nuevo)[0]

    estados = {0: "PRESENTE", 1: "TARDE", 2: "FALTA"}

    return {
        "prediccion": estados[int(pred)],
        "presente": round(prob[0] * 100, 2),
        "tarde": round(prob[1] * 100, 2),
        "falta": round(prob[2] * 100, 2)
    }