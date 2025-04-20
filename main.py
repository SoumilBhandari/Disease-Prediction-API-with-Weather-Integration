import os
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load model and encoder
model = joblib.load('model/model.joblib')
le = joblib.load('model/label_encoder.joblib')

# Define request schemas
class PredictRequest(BaseModel):
    Age: int
    Gender: str
    Temperature: float
    Humidity: float
    WindSpeed: float
    symptoms: dict

class PredictWithWeatherRequest(BaseModel):
    Age: int
    Gender: str
    WindSpeed: float
    symptoms: dict
    city: str
    api_key: str

app = FastAPI(title='Disease Prediction API')

@app.post('/predict')
def predict(req: PredictRequest):
    # Build data row
    data = req.symptoms.copy()
    data['Age'] = req.Age
    data['Gender'] = req.Gender
    data['Temperature (C)'] = req.Temperature
    data['Humidity'] = req.Humidity
    data['Wind Speed (km/h)'] = req.WindSpeed
    df = pd.DataFrame([data])
    # Predict
    pred = model.predict(df)[0]
    prognosis = le.inverse_transform([pred])[0]
    return {'prognosis': prognosis}

@app.post('/predict_with_weather')
def predict_with_weather(req: PredictWithWeatherRequest):
    # Fetch weather
    url = f'http://api.openweathermap.org/data/2.5/weather?q={req.city}&appid={req.api_key}&units=metric'
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail='Weather API error')
    w = r.json()['main']
    temp = w['temp']
    humid = w['humidity']
    # Build data
    data = req.symptoms.copy()
    data['Age'] = req.Age
    data['Gender'] = req.Gender
    data['Temperature (C)'] = temp
    data['Humidity'] = humid
    data['Wind Speed (km/h)'] = req.WindSpeed
    df = pd.DataFrame([data])
    # Predict
    pred = model.predict(df)[0]
    prognosis = le.inverse_transform([pred])[0]
    return {'prognosis': prognosis, 'weather': {'temp': temp, 'humidity': humid}}

