import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("model")

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])
    # TotalCharges is blank for brand-new customers (tenure 0), so treat it as 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def top_features(pipe: Pipeline, n: int = 10) -> list[dict]:
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    else:
        scores = abs(model.coef_[0])
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)[:n]
    return [{"feature": name.split("__", 1)[1], "importance": round(float(s), 4)} for name, s in ranked]


def main():
    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    print(f"Rows: {len(df)} | Churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    imbalance = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=imbalance, eval_metric="logloss", random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, pipes = [], {}

    for name, model in models.items():
        # Preprocessing lives inside the pipeline, so the scaler is fit on training folds only (no leakage).
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append({
            "model": name,
            "cv_f1": round(cv_f1, 3),
            "precision": round(precision_score(y_test, pred), 3),
            "recall": round(recall_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 3),
        })
        pipes[name] = pipe

    table = pd.DataFrame(results).set_index("model")
    print("\n", table.to_string(), "\n")

    # Choose using cross-validation F1 so the test set stays an unbiased final check.
    best = table["cv_f1"].idxmax()
    print(f"Best model (by CV F1): {best}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipes[best], MODEL_DIR / "churn_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"best_model": best, "results": results, "top_features": top_features(pipes[best])}, f, indent=2)
    print(f"Saved model and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
