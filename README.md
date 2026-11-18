# Impact of Social Media on Youth: A Survey-Based Data Analysis

A 2nd-year B.Tech. Data Science mini-project analyzing how social media usage patterns affect daily routines, academic focus, and sleep habits among youth.

## Project Structure
- `cleaned_survey_responses.csv` — Cleaned and anonymized dataset of 40 student/youth survey responses.
- `eda_visualize.py` — Generates summary statistics (Mean, Median, Mode) and saves exploratory charts.
- `train_model.py` — Trains Logistic Regression and Random Forest classifiers to predict study distraction and sleep disruption risk.
- `predict_impact.py` — Command-line prediction tool using the saved model (`social_media_model.joblib`).
- `requirements.txt` — Required Python libraries.

## Key Findings (N = 40)
- **Top Platform:** Instagram is the primary platform for **62.5%** (25/40) of respondents, followed by YouTube at **27.5%** (11/40).
- **Daily Screen Time:** Respondents spend an average of **3.15 hours/day** on social media, with **55.0%** spending 3–4 hours daily.
- **Academic Impact:** While **52.5%** agree that social media helps with learning (Mean = 3.38), **55.0%** report that it distracts them from studies/work (Mean = 3.62).
- **Sleep Impact:** **52.5%** report that social media negatively affects their sleep schedule (Mean = 3.40).

## How to Run Locally
```bash
pip install -r requirements.txt
python eda_visualize.py
python train_model.py
python predict_impact.py
```
