import uvicorn
import cloudpickle
from fastapi import FastAPI
from pydantic import BaseModel


class Post(BaseModel):
    postContent: str


app = FastAPI()

with open('./pipeline.pkl', 'rb') as f:
    model = cloudpickle.load(f)


@app.get("/")
def home():
    return {"Hello": "World"}


@app.post("/prediction")
def get_prediction(post: Post):
    post_data = post.dict()
    predicted_tags = model.predict([post_data['postContent']])
    return {'Tags': predicted_tags}


if __name__ == "__main__":
    uvicorn.run("fastapi:app")
