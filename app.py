import streamlit as st
import pandas as pd
from pathlib import Path
from html import escape
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AttritionAI | Smart Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Reduce the width of the entire application */
.block-container {
    max-width: 1000px;
    padding-left: 2rem;
    padding-right: 2rem;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "attrition.csv"
BANNER_PATH = ROOT / "image" / "banner.png"
if not BANNER_PATH.exists():
    BANNER_PATH = ROOT / "images" / "banner.png"

# ---------------------------------------------------------
# CSS - styling inspired by your reference screenshot
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --navy: #03102b;
  --card: #061b3b;
  --border: #164a87;
  --muted: #b8d7ff;
  --blue: #338cff;
  --purple: #962cff;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 85% 0%, rgba(36,52,130,.32), transparent 36%),
    linear-gradient(125deg, #020b20 0%, #03122e 58%, #040c29 100%);
  color: #f7f9ff;
}
.block-container { max-width: 1240px; padding: 1rem 2rem 1rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
h1,h2,h3,p,label { color: #f7f9ff; }

/* Header */
.brand { font-size: 1.65rem; line-height: 1.15; font-weight: 800; }
.brand span { color: #a56bff; }
.brand-sub { font-size: .68rem; color: #9bcaff; margin-left: 43px; }
.top-tag {
  display: inline-block; float: right; margin-top: 5px;
  padding: 7px 13px; border-radius: 30px;
  border: 1px solid #5632da; background: #16185a;
  color: #f0eaff; font-size: .68rem;
}

/* Hero banner fallback */
.hero {
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px; margin: 15px 0 16px; padding: 20px 27px;
  border-radius: 13px; border: 1px solid #4e3bdb;
  background: linear-gradient(110deg,#3021a5,#282b99 65%,#6539dc);
}
.hero-kicker { color: #e0d5ff; font-size: .95rem; font-style: italic; }
.hero-title { font-size: 1.45rem; font-weight: 800; line-height: 1.25; }
.hero-copy { color: #e0e8ff; font-size: .75rem; line-height: 1.55; }
.hero-art { font-size: 2.7rem; white-space: nowrap; }

/* Section titles */
.section-title { display:flex; align-items:center; gap:10px; font-size:1.1rem; font-weight:800; }
.section-num {
  display:inline-flex; align-items:center; justify-content:center;
  width:34px; height:34px; border-radius:50%;
  background:linear-gradient(135deg,#398bff,#a02cff); color:white;
  font-size:.82rem; font-weight:800; flex-shrink:0;
}
.section-sub { color:#b8d7ff; font-size:.72rem; margin:1px 0 10px 44px; }

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(145deg, rgba(5,29,63,.98), rgba(4,20,46,.98));
  border: 1px solid #164a87; border-radius: 11px;
  padding: 12px; box-shadow: 0 5px 18px rgba(0,0,0,.12);
}
div[data-testid="stVerticalBlockBorderWrapper"] h3 {
  font-size: 1rem; font-weight: 750; margin-bottom: .6rem;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label {
  color: #d5e6ff !important; font-size: .72rem !important;
}
div[data-testid="stNumberInput"] input {
  background: #062653 !important; color: #fff !important;
  border: 1px solid #164f91 !important; border-radius: 7px !important;
  font-size: .8rem !important;
}
div[data-testid="stNumberInput"] button {
  background: #062653 !important; color: #e7f0ff !important;
  border: 1px solid #164f91 !important;
}

/* Slider */
div[data-testid="stSlider"] [data-baseweb="slider"] > div {
  background: #152f60 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
  background: linear-gradient(90deg,#398bff,#7b35ff,#a42cff) !important;
}
div[data-testid="stSlider"] [role="slider"] {
  background: #a12cff !important; border: 3px solid #fff !important;
  box-shadow: 0 0 8px rgba(162,44,255,.5);
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
  color: #d7c3ff !important; font-weight: 700;
}
div[data-testid="stRadio"] [role="radiogroup"] { gap: 16px; }

/* Button */
.stButton > button {
  width:100%; min-height:39px; border:0 !important; border-radius:8px !important;
  background:linear-gradient(90deg,#3297ff,#6246ff,#a124ff) !important;
  color:#fff !important; font-size:.78rem; font-weight:750;
  transition: .2s ease;
}
.stButton > button:hover {
  box-shadow:0 0 18px rgba(119,64,255,.45); transform:translateY(-1px);
}

/* Prediction cards */
.result-card {
  min-height:112px; padding:15px 17px; border-radius:11px;
  background:linear-gradient(120deg,#06353d,#06223a);
  border:1px solid #00cbb4;
}
.result-icon { float:left; font-size:2.25rem; margin-right:13px; }
.result-label { color:#39e5c9; font-size:.65rem; font-weight:800; letter-spacing:.6px; }
.result-title { font-size:1.08rem; font-weight:800; margin:4px 0; }
.result-copy { color:#c6e0ff; font-size:.72rem; line-height:1.5; }
.prob-card {
  min-height:112px; padding:15px 17px; border-radius:11px;
  background:linear-gradient(120deg,#061a3b,#061631); border:1px solid #164a87;
}
.prob-layout { display:flex; align-items:center; gap:15px; }
.prob-circle {
  width:73px; height:73px; flex-shrink:0; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  background:radial-gradient(circle,#071a3b 62%,transparent 64%),
             conic-gradient(#a42cff var(--prob),#163caa 0);
  font-size:.95rem; font-weight:800;
}
.prob-heading { font-size:.8rem; font-weight:800; }
.prob-copy { color:#c3d9ff; font-size:.69rem; margin-top:5px; line-height:1.5; }
.notice {
  border:1px solid #3939a8; background:#081b50; color:#a9bcff;
  border-radius:9px; padding:9px 12px; margin-top:9px; font-size:.69rem;
}
.footer {
  text-align:center; border-top:1px solid #1d3565; padding-top:12px;
  margin-top:16px; color:#9fc4f5; font-size:.67rem;
}

/* Balloon illustrations stay at the screen edges */
.side-balloons {
  position:fixed; bottom:25%; width:58px; height:155px;
  z-index:0; pointer-events:none; opacity:.96;
}
.left-balloons { left:0; }
.right-balloons { right:0; }
.side-balloons svg { width:100%; height:100%; overflow:visible; }
@media (max-width: 900px) {
  .side-balloons { display:none; }
  .block-container { padding-left:1rem; padding-right:1rem; }
}
@media (max-width: 600px) {
  .block-container { padding-left:.65rem; padding-right:.65rem; }
  .brand { font-size:1.35rem; }
  .hero-title { font-size:1.12rem; }
  .hero-art { font-size:1.8rem; }
  .prob-circle { width:60px; height:60px; }
}
</style>
""", unsafe_allow_html=True)

# Balloon SVG decorations
balloon_svg = """
<svg viewBox="0 0 100 220" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <linearGradient id="pink" x1="0" y1="0" x2="1" y2="1">
   <stop offset="0%" stop-color="#ffc0df"/><stop offset="100%" stop-color="#f044a0"/>
  </linearGradient>
  <linearGradient id="purple" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0%" stop-color="#c5b4ff"/><stop offset="100%" stop-color="#633cff"/>
  </linearGradient>
  <linearGradient id="blue" x1="0" y1="0" x2="1" y2="1">
   <stop offset="0%" stop-color="#b4edff"/><stop offset="100%" stop-color="#3996ff"/>
  </linearGradient>
 </defs>
 <path d="M30 58 C35 110 55 130 32 192" fill="none" stroke="#985cff" stroke-width="1.5"/>
 <path d="M62 40 C48 105 80 145 56 204" fill="none" stroke="#985cff" stroke-width="1.5"/>
 <path d="M77 88 C70 130 35 150 44 211" fill="none" stroke="#985cff" stroke-width="1.5"/>
 <ellipse cx="29" cy="40" rx="19" ry="25" fill="url(#pink)"/>
 <ellipse cx="61" cy="30" rx="21" ry="29" fill="url(#purple)"/>
 <ellipse cx="76" cy="68" rx="18" ry="24" fill="url(#blue)"/>
 <ellipse cx="23" cy="31" rx="5" ry="9" fill="#fff" opacity=".28" transform="rotate(25 23 31)"/>
 <ellipse cx="55" cy="18" rx="5" ry="9" fill="#fff" opacity=".3" transform="rotate(25 55 18)"/>
 <ellipse cx="70" cy="60" rx="4" ry="8" fill="#fff" opacity=".3" transform="rotate(25 70 60)"/>
</svg>
"""
st.markdown(
    f'<div class="side-balloons left-balloons">{balloon_svg}</div>'
    f'<div class="side-balloons right-balloons"><div style="transform:scaleX(-1)">{balloon_svg}</div></div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="font-size:2rem;">👥</span>
      <div>
        <div class="brand">Attrition<span>AI</span></div>
        <div class="brand-sub" style="margin-left:0">Smart Employee Attrition Prediction</div>
      </div>
      <div style="height:35px;border-left:1px solid #17477b;margin-left:10px"></div>
      <div style="color:#8dbcf4;font-size:.67rem">Better Insights　 •　 Happier Teams　 •　 Stronger Future</div>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    st.markdown('<div class="top-tag">🧠 Machine Learning Powered</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero banner image, cropped from the reference provided.
# If removed, a CSS banner is shown instead.
# ---------------------------------------------------------
from PIL import Image

if BANNER_PATH.exists():
    banner = Image.open(BANNER_PATH)
    new_height = 400  # Change this value to adjust height
    new_width = banner.width

    banner = banner.resize((new_width, new_height))
    st.image(banner, use_container_width=True)

else:
    st.markdown("""
    <div class="hero">
      <div>
        <div class="hero-kicker">Hi! 👋</div>
        <div class="hero-title">Let's Predict Employee Attrition</div>
        <div class="hero-copy">Enter the employee details below to find out whether<br>
        they are likely to stay or leave the company.</div>
      </div>
      <div class="hero-art">🎈 🌿 👩🏻‍💻 🎈</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    if not CSV_PATH.exists():
        raise FileNotFoundError("attrition.csv must be saved beside app.py.")

    data = pd.read_csv(CSV_PATH)
    required = [
        "Age", "MonthlyIncome", "YearsAtCompany",
        "JobSatisfaction", "OverTime", "Attrition"
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(missing))

    data = data[required].dropna(subset=["Attrition"])
    X = data.drop(columns=["Attrition"])
    y = data["Attrition"].astype(str).str.strip().str.title()

    numeric = ["Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction"]
    categorical = ["OverTime"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    pipeline.fit(X, y)
    classes = list(pipeline.named_steps["model"].classes_)
    return pipeline, classes

try:
    model, classes = train_model()
except Exception as exc:
    st.error(f"Could not load the prediction model: {exc}")
    st.stop()

# ---------------------------------------------------------
# Employee profile
# ---------------------------------------------------------
st.markdown("""
<div class="section-title"><span class="section-num">01</span> Employee Profile</div>
<div class="section-sub">Provide the employee's personal and work-related details.</div>
""", unsafe_allow_html=True)

personal_col, work_col = st.columns(2, gap="medium")

with personal_col:
    with st.container(border=True):
        st.markdown("### 👤  Personal Details")
        age = st.number_input("Age (years)", min_value=18, max_value=80, value=30, step=1)
        income = st.number_input("Monthly Income (₹)", min_value=0, max_value=1000000, value=40000, step=1000)
        years = st.number_input("Years at Company", min_value=0, max_value=60, value=3, step=1)
        predict_button = st.button("✨  Analyze Employee Attrition  →", use_container_width=True)

with work_col:
    with st.container(border=True):
        st.markdown("### 💼  Work Details")
        satisfaction = st.slider(
            "Job Satisfaction", min_value=1, max_value=4, value=3,
            key="job_satisfaction"
        )
        overtime = st.radio(
            "Works Overtime?", ["No", "Yes"],
            horizontal=True, key="overtime"
        )
        st.markdown("""
        <p style="color:#a9c8ff;font-size:.69rem;margin-top:20px">
        Satisfaction scale: 1 (Low) to 4 (High).
        </p>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction result
# ---------------------------------------------------------
if predict_button:
    employee = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "JobSatisfaction": satisfaction,
        "OverTime": overtime
    }])

    prediction = str(model.predict(employee)[0]).strip().title()
    probabilities = model.predict_proba(employee)[0]
    yes_index = next(
        (i for i, label in enumerate(classes)
         if str(label).strip().lower() == "yes"),
        None
    )
    if yes_index is None:
        st.error("The Attrition column must include a 'Yes' class.")
        st.stop()

    probability = float(probabilities[yes_index]) * 100
    probability = max(0.0, min(100.0, probability))

    st.markdown("""
    <div class="section-title" style="margin-top:13px">
      <span class="section-num">02</span> Prediction Result
    </div>
    <div class="section-sub">See the model's prediction and the estimated probability of attrition.</div>
    """, unsafe_allow_html=True)

    result_col, prob_col = st.columns(2, gap="medium")

    if prediction == "Yes":
        result_icon = "⚠️"
        result_color = "#ff829a"
        result_title = "Attrition: YES"
        result_copy = "The model predicts that this employee may leave the company."
    else:
        result_icon = "🛡️"
        result_color = "#00e0b7"
        result_title = "Attrition: NO"
        result_copy = "The model predicts that this employee may stay with the company."

    with result_col:
        st.markdown(f"""
        <div class="result-card">
          <div class="result-icon">{result_icon}</div>
          <div class="result-label">MODEL PREDICTION</div>
          <div class="result-title" style="color:{result_color}">{escape(result_title)}</div>
          <div class="result-copy">{escape(result_copy)}</div>
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        st.markdown(f"""
        <div class="prob-card">
          <div class="prob-layout">
            <div class="prob-circle" style="--prob:{probability * 3.6}deg">
              {probability:.1f}%
            </div>
            <div>
              <div class="prob-heading">Attrition Probability</div>
              <div class="prob-copy">Estimated probability assigned to the Yes class.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(round(probability)), text=f"Estimated probability: {probability:.1f}%")

    st.markdown("""
    <div class="notice">
      ⓘ This is a machine-learning estimate, not a guarantee. Results depend on the quality and representativeness of the training dataset.
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
  <div style="font-weight:700;color:#bcd9ff">AttritionAI　·　Employee Analytics Project</div>
  <div style="margin-top:4px">Built with Python　·　 Streamlit　·　 Scikit-learn</div>
</div>
""", unsafe_allow_html=True)
