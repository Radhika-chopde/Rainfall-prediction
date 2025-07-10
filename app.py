from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = pickle.load(open('rainfall.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch values from form
        features = [float(request.form[key]) for key in request.form]
        
        # Convert to numpy and reshape
        input_data = np.array(features).reshape(1, -1)

        # Apply scaling if used
        input_data = scaler.transform(input_data)

        # Predict
        result = model.predict(input_data)[0]
        label = 'Yes Rain 🌧️' if result == 1 else 'No Rain ☀️'
        
        return render_template('index.html', prediction_text=f"Prediction: {label}")
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
