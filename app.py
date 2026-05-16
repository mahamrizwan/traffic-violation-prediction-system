from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("model.pkl")

time_map = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3
}

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        try:
            fine = float(request.form.get('Fine_Amount_PKR', 0))
            hour = int(request.form.get('Hour_of_Day', 0))

            time_period = request.form.get('Time_Period', 'Morning')
            time_period = time_map.get(time_period, 0)

            violation_level = int(request.form.get('Violation_Level', 0))
            payment = float(request.form.get('Payment_Compliance', 0))
            total_violations = int(request.form.get('Total_Violations', 0))
            total_unpaid = float(request.form.get('Total_Unpaid', 0))

            features = np.array([[fine, hour, time_period,
                                  violation_level, payment,
                                  total_violations, total_unpaid]])

            prediction = model.predict(features)[0]

            label_map = {
                0: "Low",
                1: "Medium",
                2: "High"
            }

            result = label_map.get(prediction, "Unknown")

            return render_template("predict.html", prediction=result)

        except Exception as e:
            return f"Prediction Error: {str(e)}"

    return render_template("predict.html", prediction=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
