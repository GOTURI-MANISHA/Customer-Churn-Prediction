import joblib

encoders = joblib.load("model/label_encoders.pkl")

for col, le in encoders.items():
    print(f"\n{col}")
    print(list(le.classes_))