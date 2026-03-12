# Flask backend for AI Study Time Recommendation System
# Uses trained models and exposes prediction API

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models
lr_model = joblib.load(os.path.join(os.path.dirname(__file__), 'lr_model.joblib'))
dt_model = joblib.load(os.path.join(os.path.dirname(__file__), 'dt_model.joblib'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting a list of dicts with keys: Hours_Studied, Previous_Scores, Motivation_Level, Sleep_Hours
    X = pd.DataFrame(data)
    # Encode Motivation_Level
    if 'Motivation_Level' in X:
        X['Motivation_Level'] = X['Motivation_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    # Predict exam scores
    exam_scores = lr_model.predict(X)
    # Predict priority (using Decision Tree, for demonstration)
    priority = dt_model.predict(X)
    # Return results
    return jsonify({
        'results': [
            {
                'input': inp,
                'predicted_exam_score': float(score),
                'priority': str(prio)
            } for inp, score, prio in zip(data, exam_scores, priority)
        ]
    })

@app.route('/')
def home():
    return 'AI Study Time Recommendation Flask Backend'

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
