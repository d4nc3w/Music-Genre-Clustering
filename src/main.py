from fastapi import FastAPI
from src.api.models import ContinueTraining
from src.api.controller import continue_train_controller

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/continue-training")
async def continue_training(training_model: ContinueTraining):
    continue_train_controller(training_model)
    return {"message": "Model successfully created"}