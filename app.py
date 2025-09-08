from flask import Flask, request, jsonify
from model import generateAI
import pickle

# Generate and load model
generateAI()
ai = pickle.load(open('ai.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "server running"

@app.route('/predict')
def predict():
    ir = request.args.get("ir")
    ir = int(ir)
    data = [[ir]]
    result = ai.predict(data)[0]
    return jsonify({"prediction": int(result)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)
