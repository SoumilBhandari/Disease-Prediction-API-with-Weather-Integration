# Disease Prediction API with Weather Integration

This repository provides:

1. **Model Training**: Train a Random Forest classifier to predict diseases based on weather and symptoms.
2. **API**: A FastAPI application with two endpoints:
   - `/predict`: Direct prediction given all features.
   - `/predict_with_weather`: Fetches current weather from OpenWeather, then predicts.

## Getting Started

### 1. Train the model

```bash
cd model
python train.py
cd ..
```

This will save `model/model.joblib` and `model/label_encoder.joblib`.

### 2. Run the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn app.main:app --reload
```

### 3. API Usage

#### Direct prediction

POST `/predict` with JSON body:

```json
{
  "Age": 30,
  "Gender": "Male",
  "Temperature": 25.0,
  "Humidity": 60.0,
  "WindSpeed": 10.0,
  "symptoms": {
    "fever": 1,
    "cough": 0,
    "...": ...
  }
}
```

#### Prediction with weather

POST `/predict_with_weather` with JSON body:

```json
{
  "Age": 30,
  "Gender": "Male",
  "WindSpeed": 10.0,
  "city": "London",
  "api_key": "YOUR_OPENWEATHER_API_KEY",
  "symptoms": {
    "fever": 1,
    "cough": 0,
    "...": ...
  }
}
```

### License

MIT License.
