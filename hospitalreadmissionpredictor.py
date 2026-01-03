import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.special import expit

st.set_page_config(page_title="Hospital Readmission Risk Intelligence",
                   page_icon="🏥",
                   layout="wide")

# ===========================
# PREMIUM UI CSS (screenshot style)
# ===========================

st.markdown("""
<style>

body {
    background: #F4F7FB;
}

.big-header {
    font-size: 38px;
    font-weight: 900;
    color: #111827;
}

.card {
    background: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0px 14px 70px rgba(0,0,0,0.06);
}

.dark-card {
    background: linear-gradient(145deg,#0F172A,#020617);
    color:white;
    padding:32px;
    border-radius:32px;
}

.metric-value {
    font-size: 44px;
    font-weight: 900;
}

.sub-text {
    color:#6B7280;
    font-size:13px;
}

.tag {
    background:#EEF2FF;
    color:#3730A3;
    padding:6px 12px;
    border-radius:15px;
    font-weight:700;
}

.bullet {
    font-weight:700;
    color:#111827;
}

.mitigation-box {
    background:white;
    border-radius:26px;
    padding:22px;
    border:1px solid #E5E7EB;
    font-style:italic;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# LOAD REAL DATA
# ===========================

@st.cache_data
def load_data():
    return pd.read_csv("FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv")

df = load_data()

# ===========================
# PREPARE MODEL DATA
# ===========================

df = df.copy()

df["Excess Readmission Ratio"] = pd.to_numeric(df["Excess Readmission Ratio"], errors="coerce")
df["Predicted Readmission Rate"] = pd.to_numeric(df["Predicted Readmission Rate"], errors="coerce")
df["Expected Readmission Rate"] = pd.to_numeric(df["Expected Readmission Rate"], errors="coerce")
df["Number of Readmissions"] = pd.to_numeric(df["Number of Readmissions"], errors="coerce")
df["Number of Discharges"] = pd.to_numeric(df["Number of Discharges"], errors="coerce")

df = df.dropna()

df["Is_Penalized"] = (df["Excess Readmission Ratio"] > 1).astype(int)

X = df[['State','Measure Name',
        'Number of Discharges',
        'Predicted Readmission Rate',
        'Expected Readmission Rate',
        'Number of Readmissions']]

y = df["Is_Penalized"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ['State','Measure Name']),
    ("num", StandardScaler(), ['Number of Discharges',
                               'Predicted Readmission Rate',
                               'Expected Readmission Rate',
                               'Number of Readmissions'])
])

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=250),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Neural Network": MLPClassifier(max_iter=400)
}

# ===========================
# UI HEADER (like screenshot)
# ===========================

st.markdown("""<div class='big-header'>
🏥 Regulatory Risk Prediction Dashboard
</div>""", unsafe_allow_html=True)

st.caption("Cycle: CMS FY-2025 Program • Real Dataset")

# ===========================
# LEFT PANEL – RISK SIMULATION
# ===========================

col1, col2, col3 = st.columns([1.4,1,1.4])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🧪 Risk Simulation Parameters")

    discharges = st.slider("Discharge Volume", 10, 5000, 800)

    pred = st.number_input("Actual Readmission Rate (%)", 5.0, 40.0, 20.0)
    exp = st.number_input("Expected Readmission Rate (%)", 5.0, 40.0, 19.5)

    readm = round(discharges * pred / 100)

    model_name = st.selectbox("Select Algorithm",
                              list(MODELS.keys()))

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# CENTER PANEL – RISK INDEX
# ===========================

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    ratio = pred/exp if exp>0 else 0

    risk_index = min(99, max(1,(ratio-0.9)*220))

    st.subheader("⚠️ Risk Index")
    st.markdown(f"<div class='metric-value'>{risk_index:.0f}%</div>", unsafe_allow_html=True)

    if risk_index < 50:
        st.success("Low Probability")
    else:
        st.error("High Probability")

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# RIGHT KPI – FINANCIAL RISK
# ===========================

with col3:
    st.markdown("<div class='dark-card'>", unsafe_allow_html=True)

    penalty_rate = max(0, ratio-1)*0.5
    financial = penalty_rate * discharges * 5000

    st.text("ESTIMATED FINANCIAL EXPOSURE")
    st.markdown(f"<div class='metric-value'>${financial:,.0f}</div>", unsafe_allow_html=True)

    st.text(f"Penalty Rate:  {penalty_rate*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# PREDICT BUTTON
# ===========================

run = st.button("🚀 EXECUTE RISK INFERENCE")

if run:

    model = MODELS[model_name]

    pipe = Pipeline([("pre", preprocessor),
                     ("clf", model)])

    pipe.fit(X, y)

    new = pd.DataFrame({
        'State':[df.iloc[0]['State']],
        'Measure Name':[df.iloc[0]['Measure Name']],
        'Number of Discharges':[discharges],
        'Predicted Readmission Rate':[pred],
        'Expected Readmission Rate':[exp],
        'Number of Readmissions':[readm]
    })

    pred_pen = pipe.predict(new)[0]

    if hasattr(pipe.named_steps['clf'], 'predict_proba'):
        prob = pipe.predict_proba(new)[0][1]
    else:
        prob = expit(pipe.decision_function(new))[0]

    st.success(f"Model Confidence: {prob*100:.1f}%")

    # ===========================
    # Explainable AI drivers
    # ===========================

    st.markdown("### 🧠 Explainable AI Drivers")

    st.markdown(f"""
**01.** Actual readmission ({pred}%) vs expected ({exp}%)  
➡ Excess Ratio = **{ratio:.2f}**

**02.** Discharge volume ({discharges}) increases dollar exposure

**03.** Penalty probability driven by readmission gap

**04.** Condition-level CMS penalty rules applied
""")

    # ===========================
    # Strategic mitigation plan
    # ===========================

    st.markdown("### ✅ Strategic Mitigation Path")

    st.markdown(f"""
<div class='mitigation-box'>

Implement a **Transition of Care program** targeting top **10% high-risk DRGs**,  
with medication reconciliation completed within **48 hours** post-discharge  
to close the current **{(pred-exp):.1f}% readmission gap**.

</div>
""", unsafe_allow_html=True)
# ===========================
# Dynamic Strategic Mitigation Path
# ===========================

st.markdown("### ✅ Strategic Mitigation Path")

gap = pred - exp

if gap <= -1:
    strategy = f"""
    **Current performance is better than national benchmark.**  
    Maintain programs with focus on:

    • Continue existing discharge planning workflows  
    • Strengthen patient follow-up only in high-risk cohorts  
    • Monitor social-determinant driven readmissions  
    • Avoid unnecessary interventions that increase cost  
    """

elif -1 < gap < 1:
    strategy = f"""
    **Very small readmission gap ({gap:.1f}%).**  
    Focus on **precision interventions** instead of large programs:

    • Nurse-led follow-up calls within 72 hours  
    • Tele-monitoring for CHF / COPD patients  
    • Flag top 5% risk score patients for review  
    • Optimize medication reconciliation quality  
    """

elif 1 <= gap < 3:
    strategy = f"""
    **Moderate risk zone (gap {gap:.1f}%).**  
    Recommend **targeted transition-of-care bundle**:

    • 48-hour post-discharge clinic visit  
    • High-risk pharmacist medication review  
    • Care coordinator assignment  
    • Discharge education with family involvement  
    """

elif 3 <= gap < 6:
    strategy = f"""
    **High penalty exposure (gap {gap:.1f}%).**  
    Recommend **hospital-wide readmission reduction plan**:

    • Observation-to-inpatient conversion audit  
    • Standardize discharge summary quality checklist  
    • Daily readmission huddle board  
    • Real-time EHR risk alerts for top decile DRGs  
    """

else:
    strategy = f"""
    🚨 **Critical Risk (gap {gap:.1f}%) – immediate intervention required.**

    • Launch enterprise readmission command center  
    • Deploy case manager at ED triage for bounce-backs  
    • Arrange home-health visits within 24 hours  
    • Create bundled care pathways for HF/COPD/AMI  
    • Report directly to Quality & Safety Committee  
    """

st.markdown(
    f"<div class='mitigation-box'>{strategy}</div>",
    unsafe_allow_html=True
)
