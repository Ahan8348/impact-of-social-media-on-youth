import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned survey dataset
df = pd.read_csv("cleaned_survey_responses.csv")
print("Dataset shape:", df.shape)

# 1. Most Used Platform breakdown
platform_counts = df["most_used_platform"].value_counts()
platform_pct = df["most_used_platform"].value_counts(normalize=True) * 100
print("\nMost Used Platform:")
print(pd.DataFrame({"Count": platform_counts, "Percentage (%)": platform_pct.round(2)}))

# 2. Daily Screen Time breakdown
screen_order = ["1–2 hours", "3–4 hours", "5–6 hours", "More than 6 hours"]
screen_counts = df["daily_screen_time"].value_counts().reindex(screen_order)
screen_pct = (screen_counts / len(df)) * 100
print("\nDaily Screen Time:")
print(pd.DataFrame({"Count": screen_counts, "Percentage (%)": screen_pct.round(2)}))

# 3. Likert Scale Summary (1 = Strongly disagree to 5 = Strongly agree)
scale_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5
}

likert_cols = [
    "q9_stay_connected",
    "q10_studies_learning",
    "q11_study_distraction",
    "q12_affects_sleep"
]

print("\nLikert Scale Summary:")
for col in likert_cols:
    scores = df[col].map(scale_map)
    print(f"{col}: Mean = {scores.mean():.2f}, Median = {scores.median():.1f}, Mode = {df[col].mode()[0]}")

# 4. Save Bar Chart of Most Used Platforms
plt.figure(figsize=(7, 4))
platform_pct.sort_values().plot(kind="barh", color=["#0A66C2", "#25D366", "#FF0000", "#E1306C"])
plt.title("Most Used Social Media Platform (N = 40)")
plt.xlabel("Percentage of Respondents (%)")
plt.tight_layout()
plt.savefig("platform_usage.png", dpi=200)
plt.close()

# 5. Save Bar Chart of Daily Screen Time
plt.figure(figsize=(7, 4))
screen_counts.plot(kind="bar", color="#1D4ED8", rot=0)
plt.title("Daily Screen Time Distribution (N = 40)")
plt.ylabel("Number of Respondents")
plt.xlabel("Daily Screen Time")
plt.tight_layout()
plt.savefig("screen_time_distribution.png", dpi=200)
plt.close()
print("\nSaved charts: platform_usage.png and screen_time_distribution.png")
