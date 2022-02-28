import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


class Post(BaseModel):
    postContent: str


app = FastAPI()

model = joblib.load('pipeline.joblib')
mlb = joblib.load('mlb.joblib')


@app.get("/")
def home():
    return {"Hello": "World"}


@app.post("/prediction")
def get_prediction(post: Post):
    post_data = post.dict()
    predicted_tags = model.predict([post_data['postContent']])
    return {'Tags': mlb.inverse_transform(predicted_tags)}


if __name__ == "__main__":
    uvicorn.run("fastapi:app")
