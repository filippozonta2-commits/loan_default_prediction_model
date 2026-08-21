from pathlib import Path
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = Path("artifacts/credit_risk_pipeline.joblib")
st.set_page_config(page_title="Credit Risk Decisioning", page_icon="💳", layout="wide")
st.title("Credit Risk Decisioning")
st.caption("Loan-level probability of default with transparent, policy-based decisions")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

model = load_model()
if model is None:
    st.warning("Model artifact not found. Run python -m src.train --data data/raw/loan.csv first.")
    st.stop()

with st.sidebar:
    st.header("Applicant")
    loan_amnt = st.number_input("Loan amount ($)", 1_000, 40_000, 15_000, 500)
    term = st.selectbox("Term", [" 36 months", " 60 months"])
    int_rate = st.slider("Interest rate (%)", 5.0, 31.0, 13.5, 0.1)
    installment = st.number_input("Monthly installment ($)", 25.0, 2_000.0, 450.0, 10.0)
    grade = st.selectbox("Grade", list("ABCDEFG"), index=2)
    emp_length = st.selectbox("Employment length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years", "n/a"], index=5)
    home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    annual_inc = st.number_input("Annual income ($)", 0.0, 2_000_000.0, 75_000.0, 1_000.0)
    verification_status = st.selectbox("Income verification", ["Not Verified", "Source Verified", "Verified"])
    purpose = st.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "car", "moving", "vacation", "other"])
    dti = st.slider("Debt-to-income ratio", 0.0, 60.0, 18.0, 0.1)
    delinq_2yrs = st.number_input("Delinquencies (2 years)", 0, 30, 0)
    inq_last_6mths = st.number_input("Credit inquiries (6 months)", 0, 20, 1)
    open_acc = st.number_input("Open accounts", 0, 80, 10)
    pub_rec = st.number_input("Public records", 0, 20, 0)
    revol_bal = st.number_input("Revolving balance ($)", 0.0, 500_000.0, 12_000.0, 500.0)
    revol_util = st.slider("Revolving utilization (%)", 0.0, 150.0, 45.0, 1.0)
    total_acc = st.number_input("Total accounts", 0, 150, 24)

row = pd.DataFrame([{"loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment, "grade": grade, "emp_length": emp_length, "home_ownership": home_ownership, "annual_inc": annual_inc, "verification_status": verification_status, "purpose": purpose, "dti": dti, "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc}])
pd_value = float(model.predict_proba(row)[0, 1])
if pd_value >= 0.35:
    decision, color = "REJECT", "#dc2626"
elif pd_value >= 0.20:
    decision, color = "MANUAL REVIEW", "#f59e0b"
else:
    decision, color = "APPROVE", "#16a34a"

c1, c2, c3 = st.columns(3)
c1.metric("Probability of default", f"{pd_value:.1%}")
c2.metric("Risk band", "High" if pd_value >= .35 else "Medium" if pd_value >= .20 else "Low")
c3.markdown(f"<h3 style='color:{color};margin-top:8px'>{decision}</h3>", unsafe_allow_html=True)
fig = go.Figure(go.Indicator(mode="gauge+number", value=pd_value * 100, number={"suffix": "%"}, title={"text": "Model-estimated default risk"}, gauge={"axis": {"range": [0, 60]}, "steps": [{"range": [0, 20], "color": "#dcfce7"}, {"range": [20, 35], "color": "#fef3c7"}, {"range": [35, 60], "color": "#fee2e2"}], "threshold": {"line": {"color": color, "width": 4}, "value": pd_value * 100}}))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)
st.subheader("Decision context")
st.dataframe(row.T.rename(columns={0: "Value"}), use_container_width=True)
st.caption("Portfolio demonstration only. Thresholds must be calibrated to expected loss, capacity, fairness requirements, and lending policy.")
