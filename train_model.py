import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv("Telecom_churn_dataset.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# --------------------------------
# 2. Prepare features and target
# --------------------------------

X = df.drop(columns=["Churn", "State"])
y = df["Churn"].astype(int)


# --------------------------------
# 3. Identify feature types
# --------------------------------

categorical_features = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# --------------------------------
# 4. Preprocessing
# --------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------
# 5. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# 6. Random Forest Model
# --------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])


# --------------------------------
# 7. Train Model
# --------------------------------

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------
# 8. Predictions
# --------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------
# 9. Evaluation
# --------------------------------

print("\nModel Performance")
print("----------------------------")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_probability))


# --------------------------------
# 10. Save Model
# --------------------------------

joblib.dump(model, "churn_model.pkl")

print("\nModel saved as churn_model.pkl")