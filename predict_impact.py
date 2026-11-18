import joblib
import pandas as pd

def main():
    artifact = joblib.load("social_media_model.joblib")
    models = artifact["models"]
    feature_cols = artifact["feature_columns"]

    print("--- Social Media Impact Predictor ---")
    print("Select daily screen time:")
    print("  1: 1-2 hours | 2: 3-4 hours | 3: 5-6 hours | 4: More than 6 hours")
    st_choice = int(input("Enter choice (1-4): ").strip())
    st_lookup = {1: 1.5, 2: 3.5, 3: 5.5, 4: 6.5}
    screen_time_hrs = st_lookup.get(st_choice, 3.5)

    platform = input("Most used platform (Instagram / YouTube / WhatsApp / LinkedIn): ").strip().lower()
    is_instagram = 1 if platform == "instagram" else 0
    is_youtube = 1 if platform == "youtube" else 0

    print("How often do you check social media?")
    print("  1: Few times a day | 2: Every 2-3 hours | 3: Every hour | 4: Almost constantly")
    checking_freq = int(input("Enter choice (1-4): ").strip())

    overuse_score = int(input("Do you spend more time online than intended? (1=Strongly Disagree to 5=Strongly Agree): ").strip())
    learning_score = int(input("Does social media help in your studies/learning? (1=Strongly Disagree to 5=Strongly Agree): ").strip())

    user_df = pd.DataFrame([{
        "screen_time_hrs": screen_time_hrs,
        "is_instagram": is_instagram,
        "is_youtube": is_youtube,
        "checking_freq": checking_freq,
        "overuse_score": overuse_score,
        "learning_score": learning_score
    }])[feature_cols]

    distract_prob = models["study_distraction"].predict_proba(user_df)[0][1] * 100
    sleep_prob = models["sleep_impact"].predict_proba(user_df)[0][1] * 100

    print("\n--- Prediction Results ---")
    print(f"Study Distraction Risk : {distract_prob:.1f}% ({'High Risk' if distract_prob >= 50 else 'Low Risk'})")
    print(f"Sleep Disruption Risk  : {sleep_prob:.1f}% ({'High Risk' if sleep_prob >= 50 else 'Low Risk'})")

if __name__ == "__main__":
    main()
