# Impact of Social Media on Youth: A Survey-Based Data Analysis

This repository contains an end-to-end data science mini-project developed for the **Essentials of Data Science** course. The project investigates how daily social media usage patterns affect young people in their studies, concentration, sleep schedules, and daily routines.

## Project Structure

| File / Folder | Purpose |
| :--- | :--- |
| `cleaned_survey_responses.csv` | Cleaned and anonymized dataset of 40 youth respondents across 17 attributes |
| `eda_visualize.py` | Python script that computes summary statistics and generates platform, screen-time, and Likert charts |
| `train_model.py` | Preprocesses data, trains Logistic Regression and Random Forest classifiers, and saves `social_media_model.joblib` |
| `predict_impact.py` | Command-line tool that loads the saved model and predicts study-distraction and sleep-impact risk |
| `requirements.txt` | Lists required Python libraries (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`) |

---

## Dataset Description

The dataset was collected via a structured Google Form (`Form Responses 1`) designed for this study. It contains **40 valid responses** across **17 attributes** covering demographic details, primary platform preferences, daily screen-time intervals, checking frequency, and 5-point Likert scale ratings ($1 = \text{Strongly Disagree}$ to $5 = \text{Strongly Agree}$).

* **Primary Platform (Q5):** Instagram (**62.5%**), YouTube (**27.5%**), WhatsApp (**7.5%**), LinkedIn (**2.5%**).
* **Daily Screen Time (Q6):** Mean of **3.15 hours/day** (55.0% spend 3–4 hours daily).
* **Target Outcomes:** Study Distraction (Q11, Mean = 3.62) and Sleep Disruption (Q12, Mean = 3.40).

---

## Git & GitHub Workflow

Version control was maintained collaboratively using Git and GitHub. The core Git commands utilized during development are summarized below:

| Git Command | Purpose |
| :--- | :--- |
| `git clone <repository-url>` | Clone the shared remote repository to local workstations |
| `git checkout -b feature/model-dev` | Create an isolated branch for developing the classification model |
| `git add . && git commit -m "msg"` | Stage and record changes with a descriptive commit message |
| `git pull origin main` | Download and merge updates from teammates |
| `git push -u origin feature/model-dev` | Push local feature branch commits to GitHub |

---

## Model Development & Results

Two supervised classification algorithms were trained to predict high study-distraction risk and sleep-disruption risk based on daily digital habits:
1. **Logistic Regression:** Regularized linear classifier with standardized features.
2. **Random Forest Classifier:** Non-linear ensemble model ($\max\_depth=3$).

### Performance Summary
* **Study Distraction Risk:** Random Forest achieves **87.5% test accuracy** and **75.0% 5-fold cross-validation accuracy**.
* **Sleep Disruption Risk:** Random Forest achieves **75.0% test accuracy** and **67.5% 5-fold cross-validation accuracy**.

---

## How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/impact-of-social-media-on-youth.git
   cd impact-of-social-media-on-youth
   pip install -r requirements.txt
   python eda_visualize.py
   python train_model.py
   python predict_impact.py
