import logging
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import List

from challenge.model import DelayModel

app = FastAPI()


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightList(BaseModel):
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(flight: FlightList) -> dict:
    try:
        target_column = None
        model = DelayModel()
        data = pd.DataFrame([dict(f) for f in flight.flights])
        if any(data['MES'] > 12):
            raise HTTPException(
                status_code=400, detail="Invalid month value in column.")
        if 'Fecha-I' in data.columns and 'Fecha-O' in data.columns:
            if any(data['Fecha-I']) and any(data['Fecha-O']):
                target_column = "delay"

        features = model.preprocess(data, target_column)
        model.fit(features)
        predictions = model.predict(features)

        return {"predict": predictions}
    except Exception as e:
        # Registra el error para poder ver los detalles en los logs del servidor
        logging.error(f"Error al procesar la solicitud: {e}")
        # Devuelve un mensaje de error descriptivo
        raise HTTPException(status_code=400, detail=str(e))
