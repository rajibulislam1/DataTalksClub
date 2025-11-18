from flask import Flask, request, jsonify
#from predict import predict

app = Flask(__name__)

@app.route('/')
def home():
    return "Logistic Regression API is running."

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()  # get JSON input
    pred_class, pred_proba = predict(data)
    return jsonify({
        'predicted_class': int(pred_class),
        'probabilities': pred_proba.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
