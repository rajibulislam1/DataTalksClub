# Load The model
import pickle
from flask import Flask, request, jsonify

output = 'logisticmodel.bin'

with open(output, 'rb') as f_in:
    dv, log = pickle.load(f_in)

app = Flask('Admission_Result')

@app.route("/predict", methods=['POST'])
def predict():
    applicants_info = request.get_json()

    X = dv.transform([applicants_info])
    y_pred = log.predict_proba(X)[0, 1]

    if y_pred >= 0.8:
        admission = True
    else:
        admission = False

    result = {
        'Admission_Class': float(y_pred),
        'Admission_Recommend': admission
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
