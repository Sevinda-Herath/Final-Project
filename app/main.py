from fastapi import FastAPI, Query
from .predict import predict_next_7_days

app = FastAPI()

@app.get("/predict")
def predict(symbol: str = Query(..., description="Stock ticker symbol")):
    result = predict_next_7_days(symbol.upper())
    return result
