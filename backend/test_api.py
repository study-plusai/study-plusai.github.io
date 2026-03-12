# Sample script to test Flask prediction API
import requests

url = 'http://127.0.0.1:5000/predict'

sample_data = [
    {
        "Hours_Studied": 20,
        "Previous_Scores": 70,
        "Motivation_Level": "Medium",
        "Sleep_Hours": 7
    },
    {
        "Hours_Studied": 15,
        "Previous_Scores": 60,
        "Motivation_Level": "High",
        "Sleep_Hours": 8
    }
]

response = requests.post(url, json=sample_data)
print(response.json())
