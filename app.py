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
