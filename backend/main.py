# Backend main entry for AI Study Time Recommendation System
# This will serve the API and handle ML model loading/prediction

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

# Input data models for new frontend
class StudyContext(BaseModel):
    sleep_hours: float = 7.0
    attendance_pct: float = 95.0
    motivation: float = 8.0
    tutoring_sessions_per_week: float = 1.0
    daily_study_limit: float = 6.0
    target_average: float = 85.0

class SubjectInput(BaseModel):
    name: str
    deadline: int
    current_score: float
    target_score: float
    complexity: int

class StudyPlanRequest(BaseModel):
    context: StudyContext
    subjects: List[SubjectInput]

# Legacy model input for compatibility
class LegacyStudyInput(BaseModel):
    subject: str
    difficulty: int
    deadline_days: int
    previous_score: float

# Load models (to be trained and saved later)
try:
    lr_model = joblib.load("backend/lr_model.joblib")
    dt_model = joblib.load("backend/dt_model.joblib")
except Exception:
    lr_model = None
    dt_model = None

@app.post("/predict")
def predict(inputs: List[LegacyStudyInput]):
    if not inputs:
        raise HTTPException(status_code=400, detail="Input list is empty")

    X = np.array([[i.difficulty, i.deadline_days, i.previous_score] for i in inputs])
    study_hours = lr_model.predict(X) if lr_model is not None else [0] * len(inputs)
    priority = dt_model.predict(X) if dt_model is not None else ["Unknown"] * len(inputs)

    return {
        "results": [
            {
                "subject": inp.subject,
                "recommended_hours": round(float(hours), 2),
                "priority": str(prio)
            }
            for inp, hours, prio in zip(inputs, study_hours, priority)
        ]
    }

@app.post("/recommend")
def recommend(payload: StudyPlanRequest):
    subjects = payload.subjects
    context = payload.context

    if not subjects:
        raise HTTPException(status_code=400, detail="No subjects provided")

    # compute per-subject weights and priority
    recommended = []
    total_weight = 0.0
    for subj in subjects:
        score_gap = max(0.0, subj.target_score - subj.current_score)
        urgency = 1.0 / max(1, subj.deadline)
        weight = (subj.complexity * 2.0) + (score_gap * 0.3) + (urgency * 15.0)
        if weight > 20:
            priority = "High"
        elif weight > 12:
            priority = "Medium"
        else:
            priority = "Low"

        rec = {
            "subject": subj.name,
            "deadline": subj.deadline,
            "current_score": subj.current_score,
            "target_score": subj.target_score,
            "complexity": subj.complexity,
            "score_gap": round(score_gap, 2),
            "weight": round(weight, 2),
            "priority": priority,
        }
        total_weight += weight
        recommended.append(rec)

    weekly_hours = context.daily_study_limit * 7.0
    if total_weight > 0.0:
        for rec in recommended:
            rec["recommended_hours"] = round((rec["weight"] / total_weight) * weekly_hours, 2)
    else:
        for rec in recommended:
            rec["recommended_hours"] = round(weekly_hours / len(recommended), 2)

    avg_current = sum(subj.current_score for subj in subjects) / len(subjects)
    predicted_class_score = min(100.0, avg_current + 5.0 + (context.sleep_hours - 7.0) * 2.0 + (context.attendance_pct - 90.0) * 0.5)

    return {
        "context": context.dict(),
        "subjects": recommended,
        "weekly_hours": round(weekly_hours, 2),
        "predicted_class_score": round(predicted_class_score, 2)
    }

@app.get("/")
def root():
    return {"message": "AI Study Time Recommendation Backend"}
