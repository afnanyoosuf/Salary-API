from fastapi import FastAPI
from pydantic import BaseModel,Field
import pickle

app = FastAPI()

with open("model.pkl","rb") as file:
    model = pickle.load(file)

class UserInput(BaseModel):
    experience : float

@app.get('/')
def home():

    return {"message":"Salary Prediction API"}

@app.post('/pedict')
def predict(data : UserInput):

    preediction = model.predict([[data.experience]])



    return {
        "predction_salary":round(float(preediction[0]),2)
    }