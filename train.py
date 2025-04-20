import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/disease_weather.csv')

# Features and target
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Identify columns
num_cols = ['Age', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
cat_cols = ['Gender']
symptom_cols = [c for c in X.columns if c not in num_cols + cat_cols]

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols + symptom_cols),
    ('cat', OneHotEncoder(), cat_cols)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,
                                                    test_size=0.2, random_state=42,
                                                    stratify=y_encoded)
pipeline.fit(X_train, y_train)

# Save model and encoder
joblib.dump(pipeline, 'model/model.joblib')
joblib.dump(le, 'model/label_encoder.joblib')
print("Model and label encoder saved.")
