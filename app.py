from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Load the scaler
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours_studied = float(request.form["Hours_Studied"])
    attendance = float(request.form["Attendance"])
    parental_involvement = float(request.form["Parental_Involvement"])
    access_to_resources = float(request.form["Access_to_Resources"])
    extracurricular_activities = float(request.form["Extracurricular_Activities"])
    sleep_hours = float(request.form["Sleep_Hours"])
    previous_scores = float(request.form["Previous_Scores"])
    motivation_level = float(request.form["Motivation_Level"])
    internet_access = float(request.form["Internet_Access"])
    tutoring_sessions = float(request.form["Tutoring_Sessions"])
    family_income = float(request.form["Family_Income"])
    teacher_quality = float(request.form["Teacher_Quality"])
    school_type = float(request.form["School_Type"])
    peer_influence = float(request.form["Peer_Influence"])
    physical_activity = float(request.form["Physical_Activity"])
    learning_disabilities = float(request.form["Learning_Disabilities"])
    parental_education_level = float(request.form["Parental_Education_Level"])
    distance_from_home = float(request.form["Distance_from_Home"])
    gender = float(request.form["Gender"])
    input_data = np.array([[
        hours_studied,
        attendance,
        parental_involvement,
        access_to_resources,
        extracurricular_activities,
        sleep_hours,
        previous_scores,
        motivation_level,
        internet_access,
        tutoring_sessions,
        family_income,
        teacher_quality,
        school_type,
        peer_influence,
        physical_activity,
        learning_disabilities,
        parental_education_level,
        distance_from_home,
        gender
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_score = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"The predicted exam score is {predicted_score} out of 100."
    )

if __name__ == "__main__":
    app.run(debug=True)