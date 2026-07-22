from fastapi import FastAPI
import joblib
import os
import pandas as pd

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API funcionando 🚀"}

BASE_DIR = os.path.dirname(__file__)
modelo = joblib.load(os.path.join(BASE_DIR, "modelo_asistencias.pkl"))

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
        "alumno_id": alumno_id,
        "prediccion": int(pred),
        "presente": round(float(prob[0])*100,2),
        "tarde": round(float(prob[1])*100,2),
        "falta": round(float(prob[2])*100,2)
    }
        "presente": float(prob[0]),
        "tarde": float(prob[1]),
        "falta": float(prob[2])
    }

@app.post("/predict-multiple")
def predict_multiple(data: dict):

    alumnos = data["alumnos"]


    df = pd.DataFrame([
        {
            "alumno_id": alumno["alumno_id"],
            "estado_num": alumno["estado"]
        }
        for alumno in alumnos
    ])


    pred = modelo.predict(df)

    prob = modelo.predict_proba(df)


    resultados=[]


    estados = {
        0:"PRESENTE",
        1:"TARDE",
        2:"FALTA"
    }


    for i, alumno in enumerate(alumnos):

        resultados.append({

            "alumno_id": alumno["alumno_id"],

            "prediccion": estados[int(pred[i])],

            "presente": round(float(prob[i][0])*100,2),

            "tarde": round(float(prob[i][1])*100,2),

            "falta": round(float(prob[i][2])*100,2)

        })


    return {
        "total":len(resultados),
        "resultados":resultados
    }
    return {
        "total":len(resultados),
        "resultados":resultados
    }
