import optuna
import pandas as pd
import sklearn.cluster
import sklearn.metrics
import sklearn.preprocessing
from optuna.study import StudyDirection
import os
import sys

try:
    from src.model_utils.utils import FEATURES
except ImportError:
    FEATURES = [
        "Beats Per Minute (BPM)", "Loudness (dB)", "Liveness",
        "Valence", "Acousticness", "Speechiness"
    ]

def load_data():
    target_path = "data/Spotify-2000.csv"
    
    if not os.path.exists(target_path):
        if os.path.exists(target_path + ".dvc"):
            print("CSV file missing but DVC file found. Run 'dvc pull'.")
        else:
            print(f"{target_path} not found.")
        sys.exit(1)
    
    print(f"Loading data from: {target_path}")
    df = pd.read_csv(target_path)
    
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)
        
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
        
    data_features = df[FEATURES]
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = scaler.fit_transform(data_features)
    return X

def objective(trial: optuna.Trial) -> float:
    X = load_data()
    
    n_clusters = trial.suggest_int("n_clusters", 2, 20)
    init_method = trial.suggest_categorical("init", ["k-means++", "random"])
    n_init = trial.suggest_int("n_init", 10, 30)
    max_iter = trial.suggest_int("max_iter", 300, 1000)

    model = sklearn.cluster.KMeans(
        n_clusters=n_clusters, 
        init=init_method, 
        n_init=n_init,
        max_iter=max_iter,
        random_state=42
    )

    try:
        labels = model.fit_predict(X)
        
        unique_labels = len(set(labels))
        if unique_labels < 2:
            score = -1.0
        else:
            score = sklearn.metrics.silhouette_score(X, labels, sample_size=2000)
            
    except Exception:
        score = -1.0

    return score

if __name__ == "__main__":
    study = optuna.create_study(
        direction=StudyDirection.MAXIMIZE, 
        storage="sqlite:///music_clustering.db",
        study_name="kmeans_optimization",
        load_if_exists=True
    )
    
    print("Starting Optuna optimization...")
    study.optimize(objective, n_trials=20)
    
    print("\nOptimization Finished")
    print(f"Best Silhouette Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
