import uvicorn
import joblib
from fastapi import FastAPI

app = FastAPI()

model = joblib.load('pipeline.joblib')
mlb = joblib.load('mlb.joblib')


@app.get("/")
def home():
    return {"Hello": "World"}


@app.get("/prediction")
def get_prediction(data: str):
    predicted_tags = model.predict([data])
    return {'Tags': mlb.inverse_transform(predicted_tags)}


if __name__ == "__main__":
    uvicorn.run("fastapi:app")
