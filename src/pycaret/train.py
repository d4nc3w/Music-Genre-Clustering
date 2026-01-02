from pycaret.clustering import *
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES = ["Beats Per Minute (BPM)", "Loudness (dB)", "Liveness",
            "Valence", "Acousticness", "Speechiness"]

def train_model(best_model_name: str, number_of_clusters:int=3):
    data = pd.read_csv(f"{BASE_DIR}/data/Spotify-2000.csv")
    if 'Index' in data.columns:
        data = data.drop('Index', axis=1)
    data = data[FEATURES]

    setup(data, normalize=True, session_id=123)
    kmeans = create_model('kmeans', num_clusters=number_of_clusters)

    scoring = pull()
    results = assign_model(kmeans)
    predictions = predict_model(kmeans, data = data.copy())


    save_model(kmeans, f"{BASE_DIR}/models/{best_model_name}")
    return scoring, predictions

#remove when cli application created
if __name__ == '__main__':
    s, p = train_model("new_best_model")
    print(f"Scoring {s}")
    print(f"Predictions {p}")