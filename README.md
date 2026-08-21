# Loan Default Prediction

End-to-end credit-risk portfolio project built on the public Lending Club dataset. It turns application-time borrower information into a probability of default, risk band, and illustrative decision.

## Highlights

- leakage-aware feature selection
- reproducible numeric and categorical preprocessing
- class-weighted logistic regression baseline
- ROC-AUC and average precision evaluation
- Streamlit decisioning dashboard

## Dataset

Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club

The raw data is excluded from Git. Install and configure the Kaggle CLI, then run:

~~~bash
pip install kaggle
bash scripts/download_data.sh
~~~

## Run locally

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --data data/raw/loan.csv
streamlit run app.py
~~~

If Kaggle produces accepted_2007_to_2018Q4.csv, pass that filename instead.

## Modeling design

Only application-time variables are used. Recoveries, payments, outstanding principal, and last-payment fields are excluded because they leak future outcomes.

The target is 0 for Fully Paid and 1 for Charged Off or Default. The baseline applies imputation, standardization, one-hot encoding, and class-weighted logistic regression. App thresholds are illustrative rather than lending policy.

## Structure

~~~text
app.py
src/data.py
src/features.py
src/train.py
scripts/download_data.sh
tests/test_features.py
~~~

## Roadmap

1. Compare gradient boosting.
2. Calibrate probabilities and optimize thresholds by expected cost.
3. Add SHAP or coefficient explanations and portfolio monitoring.
4. Add fairness and drift diagnostics.
5. Deploy after producing the model artifact.

Educational portfolio project only; not approved for real credit decisions.
