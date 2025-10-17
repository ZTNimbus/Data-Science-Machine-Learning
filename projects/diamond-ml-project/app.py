from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI()

templates = Jinja2Templates(directory="templates")

with open("diamond-model-complete.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    scaler = saved_data["scaler"]


class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float


@app.get("/", response_class=HTMLResponse)
async def home(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})


@app.post("/predict")
async def predict(features: DiamondFeatures):
    input_data = pd.DataFrame([features.model_dump()])
    print(input_data)

    for col in ["cut", "color", "clarity"]:
        encoder = encoders[col]
        input_data[col] = encoder.transform(input_data[col])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return {"predicted_price": prediction[0]}
