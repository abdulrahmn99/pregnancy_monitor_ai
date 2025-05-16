from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('pregnancy_risk_model_updated.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    bp_systolic = data['BP_Systolic']
    bp_diastolic = data['BP_Diastolic']
    mhr = data['MHR']
    fhr_baseline = data['FHR_Baseline']
    fetal_movement = data['FetalMovement']

    input_data = pd.DataFrame([[
        bp_systolic,
        bp_diastolic,
        mhr,
        fhr_baseline,
        fetal_movement
    ]], columns=['BP_Systolic', 'BP_Diastolic', 'MHR', 'FHR_Baseline', 'FetalMovement'])

    prediction = model.predict(input_data)[0]

    risk = "Risk 🚨 (Please consult your doctor)" if prediction == 1 else "Safe ✅"
    return jsonify({"result": risk})

if __name__ == '__main__':
    app.run(debug=True)
