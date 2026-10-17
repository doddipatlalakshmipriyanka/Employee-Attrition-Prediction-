
import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="AttritionAI | Smart HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# PREMIUM UI DESIGN
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #8b5cf6;
    --secondary: #6366f1;
    --background: #0b1020;
    --card: #151d32;
    --muted: #aab6cc;
    --white: #f8fafc;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #252052 0%, transparent 38%),
        radial-gradient(ellipse at 90% 30%, #17274b 0%, transparent 35%),
        #0b1020;
    font-family: 'Inter', sans-serif;
    color: var(--white);
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main header */
.hero {
    position: relative;
    overflow: hidden;
    padding: 35px;
    border-radius: 24px;
    background: linear-gradient(125deg, #29215e, #4338a9 55%, #7041c8);
    border: 1px solid rgba(196,181,253,.25);
    box-shadow: 0 15px 50px rgba(76,29,149,.23);
    margin-bottom: 28px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.20);
    padding: 7px 13px;
    border-radius: 30px;
    color: #e9ddff;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 800;
    letter-spacing: -1.2px;
    margin: 18px 0 8px 0;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 15px;
    line-height: 1.8;
}

/* Section headings */
.section-heading {
    font-size: 23px;
    font-weight: 800;
    color: #f8fafc;
    margin: 28px 0 5px 0;
}

.section-description {
    color: #aab6cc;
    font-size: 14px;
    margin-bottom: 20px;
}

/* All labels and text */
.stApp p,
.stApp label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0;
}

.stApp h2,
.stApp h3,
.stApp h4 {
    color: #f8fafc !important;
}

/* Input labels */
.stApp [data-testid="stWidgetLabel"] p {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    font-size: 14px;
}

/* Inputs */
.stApp input,
.stApp textarea {
    background: #111a2d !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: #43516b !important;
    border-radius: 10px !important;
}

/* Number input buttons */
.stApp [data-testid="stNumberInput"] button {
    background: #283650 !important;
    color: white !important;
    border-color: #43516b !important;
}

/* Select boxes */
.stApp [data-baseweb="select"] > div {
    background: #111a2d !important;
    border-color: #43516b !important;
    color: white !important;
}

.stApp [data-baseweb="select"] span {
    color: #ffffff !important;
}

.stApp [role="listbox"],
.stApp [role="option"] {
    background: #172238 !important;
    color: white !important;
}

/* Radio buttons */
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stRadio"] p {
    color: #f1f5f9 !important;
}

/* Slider */
.stApp [data-testid="stSlider"] label,
.stApp [data-testid="stSlider"] p {
    color: #f1f5f9 !important;
}

/* Help and caption text */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
    color: #aab6cc !important;
}

/* Form cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(21, 29, 50, 0.90);
    border: 1px solid #303e59;
    border-radius: 18px;
    padding: 8px;
}

/* Primary button */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 54px;
    border: 1px solid #a78bfa;
    border-radius: 13px;
    color: #ffffff !important;
    background: linear-gradient(100deg, #6366f1, #8b5cf6);
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 7px 22px rgba(99,102,241,.22);
    transition: all .2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(100deg, #7c83ff, #a78bfa);
    transform: translateY(-1px);
    color: #ffffff !important;
}

/* Result cards */
.result-card {
    background: linear-gradient(145deg, #18243b, #141b30);
    border: 1px solid #364665;
    border-radius: 20px;
    padding: 24px;
    min-height: 210px;
}

.result-label {
    font-size: 12px;
    color: #aab6cc;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.result-value {
    font-size: 29px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 12px;
}

.result-description {
    color: #cbd5e1;
    line-height: 1.7;
    font-size: 14px;
}

/* Probability bar */
.probability-track {
    height: 10px;
    border-radius: 20px;
    background: #303c55;
    overflow: hidden;
    margin-top: 18px;
}

.probability-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #818cf8, #c084fc);
}

/* Footer */
.footer {
    text-align: center;
    color: #8795ae;
    border-top: 1px solid #26334a;
    margin-top: 45px;
    padding-top: 22px;
    font-size: 12px;
}

/* Alerts */
.stApp [data-testid="stAlert"] p {
    color: inherit !important;
}

/* Divider */
.stApp hr {
    border-color: #303e59;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HERO HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ AI-POWERED HR ANALYTICS</div>
    <h1>👥 AttritionAI</h1>
    <p>
        Understand employee retention with machine learning.<br>
        Enter employee details and explore the model's prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# TRAIN MACHINE LEARNING MODEL
# ==================================================
@st.cache_resource
def train_model():

    df = pd.read_csv("attrition.csv")

    features = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "JobSatisfaction",
        "OverTime"
    ]

    required = features + ["Attrition"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in attrition.csv: {missing}"
        )

    df = df.dropna(subset=["Attrition"]).copy()
    df["Attrition"] = df["Attrition"].astype(str).str.strip()

    if not set(df["Attrition"].unique()).issubset({"Yes", "No"}):
        raise ValueError(
            "The Attrition column must contain Yes or No."
        )

    if df["Attrition"].nunique() < 2:
        raise ValueError(
            "The dataset must contain both Yes and No target values."
        )

    X = df[features]
    y = df["Attrition"]

    numeric_columns = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "JobSatisfaction"
    ]

    categorical_columns = ["OverTime"]

    preprocessing = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categorical_columns
        )
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)

    return model


try:
    with st.spinner("Preparing your AI model..."):
        model = train_model()

except Exception as e:
    st.error(f"Unable to prepare the model: {e}")
    st.stop()


# ==================================================
# EMPLOYEE INPUT FORM
# ==================================================
st.markdown("""
<div class="section-heading">01 &nbsp; Employee Profile</div>
<div class="section-description">
Provide the employee's personal and work-related details.
</div>
""", unsafe_allow_html=True)

with st.form("employee_form"):

    personal_col, work_col = st.columns(2, gap="large")

    with personal_col:
        with st.container(border=True):

            st.markdown("### 👤 Personal Details")

            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=30
            )

            income = st.number_input(
                "Monthly Income (₹)",
                min_value=0,
                max_value=10000000,
                value=40000,
                step=1000
            )

            years = st.number_input(
                "Years at Company",
                min_value=0,
                max_value=80,
                value=3
            )

    with work_col:
        with st.container(border=True):

            st.markdown("### 💼 Work Details")

            satisfaction = st.select_slider(
                "Job Satisfaction",
                options=[1, 2, 3, 4],
                value=3,
                help="1 means low satisfaction; 4 means high satisfaction."
            )

            overtime = st.radio(
                "Works Overtime?",
                options=["No", "Yes"],
                horizontal=True
            )

            st.caption(
                "Satisfaction scale: 1 = Low · 4 = High"
            )

    st.write("")

    submitted = st.form_submit_button(
        "✨  Analyze Employee Attrition"
    )


# ==================================================
# PREDICTION RESULT
# ==================================================
if submitted:

    employee = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "JobSatisfaction": satisfaction,
        "OverTime": overtime
    }])

    prediction = model.predict(employee)[0]

    probabilities = model.predict_proba(employee)[0]
    classes = model.named_steps["classifier"].classes_

    probability_yes = sum(
        probability
        for label, probability in zip(classes, probabilities)
        if label == "Yes"
    )

    st.markdown("""
    <div class="section-heading">02 &nbsp; AI Analysis</div>
    <div class="section-description">
    Review the prediction for the employee details you entered.
    </div>
    """, unsafe_allow_html=True)

    result_col, probability_col = st.columns(2, gap="large")

    with result_col:

        if prediction == "Yes":
            result_color = "#fca5a5"
            result_icon = "⚠️"
            result_title = "Attrition: YES"
            result_text = (
                "The model predicts that this employee may leave "
                "the company. This is an estimate, not a certainty."
            )
        else:
            result_color = "#86efac"
            result_icon = "🛡️"
            result_title = "Attrition: NO"
            result_text = (
                "The model predicts that this employee may stay "
                "with the company. This is an estimate, not a guarantee."
            )

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">MODEL PREDICTION</div>
            <div class="result-value" style="color:{result_color};">
                {result_icon} {result_title}
            </div>
            <div class="result-description">
                {result_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with probability_col:

        percentage = probability_yes * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">ATTRITION PROBABILITY</div>
            <div class="result-value" style="color:#c4b5fd;">
                {percentage:.1f}%
            </div>
            <div class="result-description">
                Estimated probability assigned to the Yes class.
            </div>
            <div class="probability-track">
                <div class="probability-fill"
                     style="width:{percentage:.2f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if prediction == "Yes":
        st.warning(
            "Interpretation: The model selected Yes as its predicted "
            "class. Review the result carefully; it does not establish "
            "that the employee will leave."
        )
    else:
        st.success(
            "Interpretation: The model selected No as its predicted "
            "class. This does not guarantee that the employee will stay."
        )

    with st.expander("📋 View submitted employee details"):
        st.dataframe(employee, use_container_width=True)

    st.caption(
        "Educational project only. Predictions depend on the training "
        "data and should not be used alone for employment decisions."
    )


# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    <b>AttritionAI</b> · Employee Analytics Dashboard<br><br>
    Built with Python · Streamlit · Scikit-learn
</div>
""", unsafe_allow_html=True)
