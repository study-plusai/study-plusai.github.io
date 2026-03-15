# Flask backend for AI Study Time Recommendation System
# Uses trained models and exposes prediction API

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Load models if available
try:
    lr_model = joblib.load(os.path.join(os.path.dirname(__file__), 'lr_model.joblib'))
    dt_model = joblib.load(os.path.join(os.path.dirname(__file__), 'dt_model.joblib'))
except Exception:
    lr_model = None
    dt_model = None

# Helper to compute priority and study plan similar to frontend logic

def compute_subject_recommendation(subject, context):
    current = float(subject.get('currentScore', 0))
    target = float(subject.get('targetScore', current))
    complexity = float(subject.get('complexity', 1))
    deadline = float(subject.get('deadline', 1))
    deadline = max(deadline, 1)

    score_gap = max(0, target - current)
    urgency = 1.0 / deadline

    weight = (complexity * 2) + (score_gap * 0.3) + (urgency * 15)

    if weight > 20:
        priority = 'High'
    elif weight > 12:
        priority = 'Medium'
    else:
        priority = 'Low'

    return {
        'subject': subject.get('name', 'Unknown'),
        'deadline': deadline,
        'currentScore': current,
        'targetScore': target,
        'complexity': complexity,
        'weight': round(weight, 2),
        'priority': priority,
        'scoreGap': round(score_gap, 2)
    }


@app.route('/predict', methods=['POST'])
def predict():
    # Backwards-compatible endpoint for legacy payloads
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Legacy compatibility: list of training rows (Hours_Studied, Previous_Scores, Motivation_Level, Sleep_Hours)
    if isinstance(data, list) and all(isinstance(x, dict) and 'Hours_Studied' in x for x in data):
        # non-ML fallback if model missing
        try:
            X = pd.DataFrame(data)
            if 'Motivation_Level' in X:
                X['Motivation_Level'] = X['Motivation_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
            if lr_model is not None:
                exam_scores = lr_model.predict(X)
            else:
                exam_scores = [0] * len(X)
            if dt_model is not None:
                priority = dt_model.predict(X)
            else:
                priority = ['Unknown'] * len(X)

            return jsonify({'results': [
                {'input': inp, 'predicted_exam_score': float(score), 'priority': str(prio)}
                for inp, score, prio in zip(data, exam_scores, priority)
            ]})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # New payload format
    if isinstance(data, dict) and 'subjects' in data and 'context' in data:
        context = data.get('context', {})
        subjects = data.get('subjects', [])

        recommended = []
        # calculate base statistics
        total_weight = 0.0
        for subj in subjects:
            rec = compute_subject_recommendation(subj, context)
            total_weight += rec['weight']
            recommended.append(rec)

        weekly_hours = max(0, float(context.get('dailyStudyLimit', context.get('daily_study_limit', 6)))) * 7

        if total_weight > 0:
            for rec in recommended:
                rec['recommendedHours'] = round((rec['weight'] / total_weight) * weekly_hours, 2)
        else:
            for rec in recommended:
                rec['recommendedHours'] = round(weekly_hours / len(recommended) if recommended else 0, 2)

        # predicted class average score from context + current
        avg_current = (sum([float(s.get('currentScore', 0)) for s in subjects]) / len(subjects)) if subjects else 0
        sleep = float(context.get('sleepHours', context.get('sleep_hours', 7)))
        attendance = float(context.get('attendance', context.get('attendance_pct', 90)))
        predicted_score = min(100, avg_current + 5 + (sleep - 7) * 2 + (attendance - 90) * 0.5)

        return jsonify({
            'context': context,
            'subjects': recommended,
            'weeklyHours': round(weekly_hours, 2),
            'predictedClassScore': round(predicted_score, 2)
        })

    return jsonify({'error': 'Invalid payload format'}), 400


@app.route('/')
def home():
    return {'message': 'AI Study Time Recommendation Flask Backend'}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
