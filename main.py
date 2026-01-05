from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ed_forecaster import EDVisitForecaster
import numpy as np

app = FastAPI()
forecaster = EDVisitForecaster(window_size=24)
forecaster.load_artifacts('ed_model.h5', 'scaler.pkl')

class PredictRequest(BaseModel):
    last_window: list

@app.get("/")
def read_root():
    return {"message": "ED Visit Forecaster API is running."}

@app.post("/predict")
def predict(request: PredictRequest):
    if len(request.last_window) != forecaster.window_size:
        raise HTTPException(status_code=400, detail=f"last_window must have {forecaster.window_size} values")
    prediction = forecaster.predict_next_hour(request.last_window)
    return {"predicted_next_hour_visits": prediction}