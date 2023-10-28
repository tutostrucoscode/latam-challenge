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
    model = DelayModel()
    data = pd.DataFrame([dict(f) for f in flight.flights])
    if any(data['MES'] > 12):
        raise HTTPException(
            status_code=400, detail="Invalid month value in column.")
    features = model.preprocess(data)
    model.fit(features)
    predictions = model.predict(features)

    return {"predict": predictions}
