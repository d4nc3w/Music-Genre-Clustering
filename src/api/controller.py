import os
from src.api.models import ContinueTraining
from pathlib import Path
from src.model_utils.utils import train_model

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = f"{BASE_DIR}/data"
MODELS_DIR = f"{BASE_DIR}/models"
FEATURES = "Index,Title,Artist,Top Genre,Year,Beats Per Minute (BPM),Energy,Danceability,Loudness (dB),Liveness,Valence,Length (Duration),Acousticness,Speechiness,Popularity"
def continue_train_controller(training_model: ContinueTraining):
    with open(f"{DATA_DIR}/tmp.csv", 'w') as file:
        file.write(FEATURES + "\n")
        for line in training_model.train_input:
            file.write(line + "\n")

    train_model(MODELS_DIR, training_model.new_model_name)
    os.remove(f"{DATA_DIR}/tmp.csv")