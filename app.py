from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'house_price_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Please run model_development.py first.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            if model is None:
                raise ValueError("Model is not loaded.")
                
            # Extract features from form
            features = [
                float(request.form['overall_qual']),
                float(request.form['gr_liv_area']),
                float(request.form['total_bsmt_sf']),
                float(request.form['garage_cars']),
                float(request.form['full_bath']),
                int(request.form['year_built'])
            ]
            
            # Predict
            final_features = [np.array(features)]
            prediction_val = model.predict(final_features)[0]
            prediction = f"${prediction_val:,.2f}"
            
        except ValueError as e:
            error = f"Invalid input: {str(e)}"
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
