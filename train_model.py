import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load cleaned dataset
df = pd.read_csv("cleaned_survey_responses.csv")

# 2. Encode features into numbers
screen_time_map = {
    "1–2 hours": 1.5,
    "3–4 hours": 3.5,
    "5–6 hours": 5.5,
    "More than 6 hours": 6.5
}
checking_map = {
    "Few times a day": 1,
    "Every 2–3 hours": 2,
    "Every hour": 3,
    "Almost constantly": 4
}
likert_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5
}

X = pd.DataFrame({
    "screen_time_hrs": df["daily_screen_time"].map(screen_time_map),
    "is_instagram": (df["most_used_platform"] == "Instagram").astype(int),
    "is_youtube": (df["most_used_platform"] == "YouTube").astype(int),
    "checking_freq": df["checking_frequency"].map(checking_map),
    "overuse_score": df["spend_more_time_than_intended"].map(likert_map),
    "learning_score": df["q10_studies_learning"].map(likert_map)
})

# Target 1: High Study Distraction (1 if Agree/Strongly Agree, else 0)
y_distraction = (df["q11_study_distraction"].map(likert_map) >= 4).astype(int)

# Target 2: Sleep Disruption (1 if Agree/Strongly Agree, else 0)
y_sleep = (df["q12_affects_sleep"].map(likert_map) >= 4).astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
saved_models = {}

for target_name, y in [("study_distraction", y_distraction), ("sleep_impact", y_sleep)]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42))
    ])
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

    print(f"=== Results for {target_name} ===")
    for model_name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        prec = precision_score(y_test, preds, zero_division=0) * 100
        rec = recall_score(y_test, preds, zero_division=0) * 100
        f1 = f1_score(y_test, preds, zero_division=0) * 100
        cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean() * 100
        print(f"{model_name}: Test Acc={acc:.1f}%, Precision={prec:.1f}%, Recall={rec:.1f}%, F1={f1:.1f}%, 5-Fold CV={cv_acc:.1f}%")

    # Refit Random Forest on full dataset and store for prediction script
    rf_model.fit(X, y)
    saved_models[target_name] = rf_model

artifact = {
    "models": saved_models,
    "feature_columns": list(X.columns)
}
joblib.dump(artifact, "social_media_model.joblib")
print("\nSaved trained models to social_media_model.joblib")
