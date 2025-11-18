import joblib
import pandas as pd

# Load model and vectorizer
log = joblib.load('models/logistic_model.pkl')
dv = joblib.load('models/dv.pkl')

def predict(input_data: dict):
    """
    input_data: dict with features
    returns: predicted class and probabilities
    """
    X = dv.transform([input_data])  # single record
    pred_class = log.predict(X)[0]
    pred_proba = log.predict_proba(X)[0]
    return pred_class, pred_proba

input_data = {'parents': 'great_pret', 
              'has_nurs':'critical', 
              'form':'complete', 
              'children':3, 
              'housing':'convenient', 
              'finance':'convenient',
              'social':'nonprob', 
              'health':'priority'}

# 👉 Call the function
pred_class, pred_proba = predict(input_data)

print("Predicted class:", pred_class)
print("Prediction probabilities:", pred_proba)


