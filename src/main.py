from fastapi import FastAPI
from httpcore import Request
from lark.grammar import NonTerminal

from src.api.models import ContinueTraining
from src.api.controller import continue_train_controller, list_models_controller
from src.api.models import PredictionInput
from src.api.controller import predict_controller
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.exception_handler(RequestValidationError)
async def model_validation_exception(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"message": "input correct data"},
    )
@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "something went wrong"}
    )

@app.exception_handler(HTTPException)
async def http_exception(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@app.post("/continue-train")
async def continue_training(training_model: ContinueTraining):
    return continue_train_controller(training_model)

@app.post("/predict")
async def predict(input_data: PredictionInput):
    predictions = predict_controller(input_data)
    return {"predictions": predictions}

@app.get("/models", response_model=list[str])
async def models():
    return list_models_controller()