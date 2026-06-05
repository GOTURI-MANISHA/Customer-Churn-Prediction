from django.shortcuts import render
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/churn_model.pkl")
encoders = joblib.load("model/label_encoders.pkl")


def home(request):

    result = None
    stay_prob = None
    churn_prob = None

    if request.method == "POST":

        data = {
            "gender": request.POST["gender"],
            "SeniorCitizen": int(request.POST["SeniorCitizen"]),
            "Partner": request.POST["Partner"],
            "Dependents": request.POST["Dependents"],
            "tenure": int(request.POST["tenure"]),
            "PhoneService": request.POST["PhoneService"],
            "MultipleLines": request.POST["MultipleLines"],
            "InternetService": request.POST["InternetService"],
            "OnlineSecurity": request.POST["OnlineSecurity"],
            "OnlineBackup": request.POST["OnlineBackup"],
            "DeviceProtection": request.POST["DeviceProtection"],
            "TechSupport": request.POST["TechSupport"],
            "StreamingTV": request.POST["StreamingTV"],
            "StreamingMovies": request.POST["StreamingMovies"],
            "Contract": request.POST["Contract"],
            "PaperlessBilling": request.POST["PaperlessBilling"],
            "PaymentMethod": request.POST["PaymentMethod"],
            "MonthlyCharges": float(request.POST["MonthlyCharges"]),
            "TotalCharges": float(request.POST["TotalCharges"]),
        }

        # Encode categorical columns
        for col in encoders:
            if col != "Churn":
                data[col] = encoders[col].transform([data[col]])[0]

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(df)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(df)[0]

        stay_prob = min(
            round(probabilities[0] * 100, 2),
            99.9
        )

        churn_prob = min(
            round(probabilities[1] * 100, 2),
            99.9
        )

        # Result text
        result = (
            "Customer Will Churn"
            if prediction == 1
            else "Customer Will Stay"
        )

    return render(
        request,
        "home.html",
        {
            "result": result,
            "stay_prob": stay_prob,
            "churn_prob": churn_prob,
        },
    )