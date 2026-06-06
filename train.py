import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("salary.csv")

X = df[["experience"]]
y = df["salary"]

model = LinearRegression()
model.fit(X,y)

with open ("model.pkl","wb") as file:
    pickle.dump(model,file)

print("Model Saved Successfully")
