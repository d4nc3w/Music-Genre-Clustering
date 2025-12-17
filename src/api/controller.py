import os
from src.api.models import ContinueTraining, PredictionInput
from pathlib import Path
from src.model_utils.utils import train_model, predict_entry

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = f"{BASE_DIR}/data"
MODELS_DIR = f"{BASE_DIR}/models"
FEATURES = "Index,Title,Artist,Top Genre,Year,Beats Per Minute (BPM),Energy,Danceability,Loudness (dB),Liveness,Valence,Length (Duration),Acousticness,Speechiness,Popularity"

def continue_train_controller(training_model: ContinueTraining):
    print(DATA_DIR)
    print(MODELS_DIR)
    print(BASE_DIR)
    with open(f"{DATA_DIR}/tmp.csv", 'w') as file:
        file.write(FEATURES + "\n")
        for line in training_model.train_input:
            file.write(line + "\n")

    train_model(MODELS_DIR, training_model.new_model_name)
    os.remove(f"{DATA_DIR}/tmp.csv")

def predict_controller(prediction_input: PredictionInput):
    tmp_filename = f"tmp_predict_{prediction_input.model_name}.csv"
    tmp_file_path = f"{DATA_DIR}/{tmp_filename}"
    with open(tmp_file_path, 'w') as file:
        file.write(FEATURES + "\n")
        for line in prediction_input.input_data:
            file.write(line + "\n")

    predictions = predict_entry(tmp_file_path, MODELS_DIR, prediction_input.model_name)
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    return predictions