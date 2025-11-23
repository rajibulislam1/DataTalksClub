
import requests

url = 'http://localhost:9696/predict'

# 
admission_info = {
    "parents":"usual", 
    "has_nurs":"proper",
    "form":"completed",
    "children":3,
    "housing":"convenient", 
    "finance":"convenient",
    "social":"slightly_prob", 
    "health":"priority"
}


response = requests.post(url, json=admission_info).json()
print(response)


