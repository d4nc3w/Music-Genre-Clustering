import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os

FEATURES = ["Beats Per Minute (BPM)", "Loudness (dB)", "Liveness",
            "Valence", "Acousticness", "Speechiness"]

def train_model(data: pd.DataFrame, model_directory_path: str, model_name: str):
    # Drop the 'Index' column as it is not useful
    if 'Index' in data.columns:
        data = data.drop('Index', axis=1)
    # Select relevant audio features for clustering
    data_features = data[FEATURES]
    # Normalize the features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_features)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    quality_metrics = {
        "inertia": kmeans.inertia_,
        "silhoutte": silhouette_score(data_scaled, labels),
        "calinski": calinski_harabasz_score(data_scaled, labels),
        "davies": davies_bouldin_score(data_scaled, labels)
    }

    joblib.dump(kmeans, f"{model_directory_path}/{model_name}.joblib")
    return quality_metrics

def predict_entry(data: pd.DataFrame, model_dir: str, model_name: str):
    if 'Index' in data.columns:
        data = data.drop('Index', axis=1)

    check_data(data)

    data_features = data[FEATURES]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_features)

    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found in {model_dir}")

    kmeans = joblib.load(model_path)
    predictions = kmeans.predict(data_scaled)
    return predictions.tolist()

def check_data(data: pd.DataFrame):
    missing_cols = [col for col in FEATURES if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")