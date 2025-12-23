from pydantic import BaseModel

class TrainInput(BaseModel):
    index: int
    title: str
    artist: str
    top_genre: str
    year: int
    beats_per_minute: int
    energy: int
    danceability: int
    loudness: int
    liveness: int
    valence: int
    length: int
    acousticness: int
    speechiness: int
    popularity: int

#only new_model_name because it is impossible to continue training on the KMeans
class ContinueTraining(BaseModel):
    new_model_name: str
    train_input: list[TrainInput]

class PredictionInput(BaseModel):
    model_name: str
    input_data: list[str]
