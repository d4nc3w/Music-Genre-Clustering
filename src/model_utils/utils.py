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

def predict_entry(data_path: str, model_dir: str, model_name: str):
    data = pd.read_csv(data_path)
    if 'Index' in data.columns:
        data = data.drop('Index', axis=1)

    features = ["Beats Per Minute (BPM)", "Loudness (dB)", "Liveness",
                "Valence", "Acousticness", "Speechiness"]

    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")

    data_features = data[features]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_features)

    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found in {model_dir}")

    kmeans = joblib.load(model_path)
    predictions = kmeans.predict(data_scaled)
    return predictions.tolist()