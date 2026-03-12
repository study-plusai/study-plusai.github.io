# Backend main entry for AI Study Time Recommendation System
# This will serve the API and handle ML model loading/prediction

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model for input
class StudyInput(BaseModel):
    subject: str
    difficulty: int
    deadline_days: int
    previous_score: float

# Load models (to be trained and saved later)
try:
    lr_model = joblib.load("backend/lr_model.joblib")
    dt_model = joblib.load("backend/dt_model.joblib")
except:
    lr_model = None
    dt_model = None

@app.post("/predict")
def predict(inputs: list[StudyInput]):
    # Prepare input features
    X = np.array([[i.difficulty, i.deadline_days, i.previous_score] for i in inputs])
    # Predict study hours
    study_hours = lr_model.predict(X) if lr_model else [0]*len(inputs)
    # Predict priority
    priority = dt_model.predict(X) if dt_model else ["Unknown"]*len(inputs)
    # Return results
    return {
        "results": [
            {
                "subject": inp.subject,
                "recommended_hours": round(hours, 2),
                "priority": prio
            } for inp, hours, prio in zip(inputs, study_hours, priority)
        ]
    }

@app.get("/")
def root():
    return {"message": "AI Study Time Recommendation Backend"}
