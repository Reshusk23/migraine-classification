import numpy as np
from flask import Flask, json, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/inputs")
def inputs():
    return render_template('headacheform.html')

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    with open('model_files/migraine_model2022.sav', 'rb') as f:
        model = pickle.load(f)
        f.close()
        
    prediction = model.predict(final_features)
    return render_template('result.html', prediction=prediction[0])



if __name__ == "__main__":
    app.run(debug=True)