import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Dataset Shape:", df.shape)

# Drop customerID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Find all categorical columns
categorical_cols = df.select_dtypes(
    include=["object"]
).columns

print("\nCategorical Columns:")
print(list(categorical_cols))

# Encode categorical columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    label_encoders[col] = le

print("\nData Types After Encoding:")
print(df.dtypes)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Verify no object columns remain
print("\nRemaining Object Columns:")
print(X.select_dtypes(include=["object"]).columns)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining Model...")
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Save Model

joblib.dump(
    list(X.columns),
    "model/feature_columns.pkl"
)

joblib.dump(
    model,
    "model/churn_model.pkl"
)

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)

print("\nModel Saved Successfully!")