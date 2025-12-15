import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib

def train_model(file_path: str, model_name: str):
    # Load dataset
    data = pd.read_csv("data/Spotify-2000.csv")
    # Drop the 'Index' column as it is not useful
    if 'Index' in data.columns:
        data = data.drop('Index', axis=1)
    # Select relevant audio features for clustering
    features = ["Beats Per Minute (BPM)", "Loudness (dB)", "Liveness",
                "Valence", "Acousticness", "Speechiness"]
    data_features = data[features]
    # Normalize the features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_features)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    joblib.dump(kmeans, f"{file_path}/{model_name}.joblib")