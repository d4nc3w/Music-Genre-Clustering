from fastapi import HTTPException
from src.api.models import ContinueTraining, PredictionInput
from pathlib import Path
from src.model_utils.utils import train_model, predict_entry, list_models
from pandas import DataFrame

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = f"{BASE_DIR}/data"
MODELS_DIR = f"{BASE_DIR}/models"
FEATURES = "Index,Title,Artist,Top Genre,Year,Beats Per Minute (BPM),Energy,Danceability,Loudness (dB),Liveness,Valence,Length (Duration),Acousticness,Speechiness,Popularity"
COLUMN_MAP = {
    "index": "Index",
    "title": "Title",
    "artist": "Artist",
    "top_genre": "Top Genre",
    "year": "Year",
    "beats_per_minute": "Beats Per Minute (BPM)",
    "energy": "Energy",
    "danceability": "Danceability",
    "loudness": "Loudness (dB)",
    "liveness": "Liveness",
    "valence": "Valence",
    "length": "Length (Duration)",
    "acousticness": "Acousticness",
    "speechiness": "Speechiness",
    "popularity": "Popularity",
}

def continue_train_controller(training_model: ContinueTraining):
    if _check_model_exists(training_model.new_model_name):
        raise HTTPException(status_code=400, detail=f"Model {training_model.new_model_name} already exists")

    if len(training_model.train_input) == 0:
        raise HTTPException(status_code=400, detail=f"No training data provided")

    df = DataFrame([item.model_dump() for item in training_model.train_input])
    df.rename(columns=COLUMN_MAP, inplace=True)

    return train_model(df, MODELS_DIR, training_model.new_model_name)

def predict_controller(prediction_input: PredictionInput):
    if not _check_model_exists(prediction_input.model_name):
        raise HTTPException(status_code=404, detail=f"Model with name {prediction_input.model_name} not found.")
    if len(prediction_input.input_data) == 0:
        raise HTTPException(status_code=404, detail=f"No data to predict provided")

    df = DataFrame([item.model_dump() for item in prediction_input.input_data])
    df.rename(columns=COLUMN_MAP, inplace=True)

    predictions = predict_entry(df, MODELS_DIR, prediction_input.model_name)
    return predictions

def list_models_controller() -> list[str]:
    models_path = Path(MODELS_DIR)
    return list_models(models_path)

def _check_model_exists(model_name: str) -> bool:
    models = list_models(Path(MODELS_DIR))
    return model_name in models
