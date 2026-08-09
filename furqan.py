"""
IrisAI – Premium ML Classification Platform
Luxury / Trillionaire-Tech UI
Built with Streamlit and the classic Iris dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IrisAI | Premium ML Platform",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LUXURY THEMES
# ============================================================

LUXURY_THEMES = {
    "Royal Gold": {
        "accent": "#D4AF37",
        "accent_light": "#F4D06F",
        "accent_bright": "#FFE9A3",
        "glow": "212,175,55",
        "secondary": "#B8860B",
        "plot": ["#D4AF37", "#F4D06F", "#FFE9A3"],
    },
    "Emerald Elite": {
        "accent": "#00C896",
        "accent_light": "#4DE1B5",
        "accent_bright": "#A6FFE5",
        "glow": "0,200,150",
        "secondary": "#008F6B",
        "plot": ["#00C896", "#4DE1B5", "#A6FFE5"],
    },
    "Sapphire Royal": {
        "accent": "#4D9FFF",
        "accent_light": "#82BFFF",
        "accent_bright": "#C7E3FF",
        "glow": "77,159,255",
        "secondary": "#2864C7",
        "plot": ["#4D9FFF", "#82BFFF", "#C7E3FF"],
    },
    "Imperial Purple": {
        "accent": "#9B6CFF",
        "accent_light": "#C0A3FF",
        "accent_bright": "#E2D5FF",
        "glow": "155,108,255",
        "secondary": "#6840C9",
        "plot": ["#9B6CFF", "#C0A3FF", "#E2D5FF"],
    },
    "Platinum Silver": {
        "accent": "#C9D1D9",
        "accent_light": "#F0F3F6",
        "accent_bright": "#FFFFFF",
        "glow": "201,209,217",
        "secondary": "#8D98A5",
        "plot": ["#C9D1D9", "#F0F3F6", "#FFFFFF"],
    },
}


# ============================================================
# PASSWORD PROTECTION
# ============================================================

def check_password():

    def password_entered():

        try:
            correct_password = st.secrets["password"]
        except Exception:
            st.session_state["password_error"] = True
            st.session_state["password_correct"] = False
            return

        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:

        st.markdown(
            """
            <div class="login-wrapper">
                <div class="login-logo">🌸</div>
                <div class="login-title">IrisAI</div>
                <div class="login-subtitle">
                    PREMIUM MACHINE LEARNING PLATFORM
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_input(
            "🔐 Enter Access Password",
            type="password",
            on_change=password_entered,
            key="password",
        )

        if st.session_state.get("password_error"):
            st.error(
                "⚠️ Password is not configured. "
                "Add password = \"YourPassword\" to .streamlit/secrets.toml"
            )

        return False

    elif not st.session_state["password_correct"]:

        st.markdown(
            """
            <div class="login-wrapper">
                <div class="login-logo">🔒</div>
                <div class="login-title">IrisAI</div>
                <div class="login-subtitle">
                    AUTHENTICATION REQUIRED
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_input(
            "🔐 Enter Access Password",
            type="password",
            on_change=password_entered,
            key="password",
        )

        st.error("❌ Incorrect password")

        return False

    return True


# ============================================================
# INITIAL SESSION STATE
# ============================================================

if "font_choice" not in st.session_state:
    st.session_state.font_choice = "Inter"

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Royal Gold"

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# FONT OPTIONS
# ============================================================

FONT_OPTIONS = {
    "Inter": "'Inter', sans-serif",
    "Poppins": "'Poppins', sans-serif",
    "Montserrat": "'Montserrat', sans-serif",
    "Roboto": "'Roboto', sans-serif",
    "Playfair Display": "'Playfair Display', serif",
    "Cormorant Garamond": "'Cormorant Garamond', serif",
    "Space Grotesk": "'Space Grotesk', sans-serif",
    "Outfit": "'Outfit', sans-serif",
}


# ============================================================
# LOAD CSS
# ============================================================

def load_css(font_name, theme_name):

    font_family = FONT_OPTIONS[font_name]
    theme = LUXURY_THEMES[theme_name]

    accent = theme["accent"]
    accent_light = theme["accent_light"]
    accent_bright = theme["accent_bright"]
    glow = theme["glow"]
    secondary = theme["secondary"]

    st.markdown(
        f"""
        <style>

        @import url(
        'https://fonts.googleapis.com/css2?
        family=Inter:wght@300;400;500;600;700;800;900&
        family=Poppins:wght@400;500;600;700;800;900&
        family=Montserrat:wght@400;500;600;700;800;900&
        family=Roboto:wght@400;500;700;900&
        family=Playfair+Display:wght@400;500;600;700&
        family=Cormorant+Garamond:wght@400;500;600;700&
        family=Space+Grotesk:wght@400;500;600;700&
        family=Outfit:wght@400;500;600;700;800&
        display=swap');

        :root {{
            --bg: #050608;
            --bg2: #080A0E;
            --panel: #101319;
            --panel2: #151922;
            --panel3: #1A1F29;

            --accent: {accent};
            --accent-light: {accent_light};
            --accent-bright: {accent_bright};
            --secondary: {secondary};

            --glow: {glow};

            --text: #F8FAFC;
            --text-soft: #E5E9EF;
            --muted: #AAB3C0;
            --muted2: #7D8796;

            --success: #43D17A;
            --danger: #FF5C65;
            --warning: #FFCA55;
            --info: #5EA7FF;
        }}

        html,
        body,
        .stApp,
        [class*="css"],
        button,
        input,
        textarea,
        select {{
            font-family: {font_family} !important;
        }}

        .stApp {{
            background:
                radial-gradient(
                    circle at 5% 0%,
                    rgba({glow}, 0.13),
                    transparent 27%
                ),
                radial-gradient(
                    circle at 95% 8%,
                    rgba(90,110,190,0.10),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 50% 100%,
                    rgba({glow},0.05),
                    transparent 35%
                ),
                linear-gradient(
                    135deg,
                    #050608 0%,
                    #080A0E 48%,
                    #050608 100%
                ) !important;

            color: var(--text) !important;
        }}

        .main {{
            background: transparent !important;
        }}

        .block-container {{
            max-width: 1550px !important;
            padding-top: 2rem !important;
            padding-bottom: 5rem !important;
        }}

        /* ====================================================
           GLOBAL VISIBILITY
           ==================================================== */

        h1, h2, h3, h4, h5, h6,
        p, span, label, li, strong,
        [data-testid="stMarkdownContainer"] {{
            color: var(--text) !important;
        }}

        p {{
            color: var(--muted) !important;
            line-height: 1.7;
        }}

        h1 {{
            font-weight: 800 !important;
            letter-spacing: -0.035em;
        }}

        h2 {{
            font-weight: 750 !important;
            letter-spacing: -0.02em;
        }}

        h3, h4 {{
            font-weight: 700 !important;
        }}

        /* ====================================================
           SIDEBAR
           ==================================================== */

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    #0D1016 0%,
                    #080A0E 55%,
                    #06070A 100%
                ) !important;

            border-right:
                1px solid rgba({glow},0.20) !important;

            box-shadow:
                15px 0 50px rgba(0,0,0,0.30);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text) !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: rgba({glow},0.18) !important;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            color: #E9EDF3 !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            color: var(--accent-light) !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] {{
            background: #141821 !important;
        }}

        /* ====================================================
           SIDEBAR BRAND
           ==================================================== */

        .sidebar-brand {{
            text-align: center;
            padding: 10px 5px 5px 5px;
        }}

        .sidebar-brand-icon {{
            font-size: 3rem;
            filter:
                drop-shadow(0 0 15px rgba({glow},0.50));
        }}

        .sidebar-brand-title {{
            font-size: 1.8rem;
            font-weight: 900;
            color: var(--accent-light) !important;
            letter-spacing: -0.04em;
        }}

        .sidebar-brand-sub {{
            font-size: 0.65rem;
            letter-spacing: 0.16em;
            color: var(--muted2) !important;
            font-weight: 700;
        }}

        /* ====================================================
           LOGIN
           ==================================================== */

        .login-wrapper {{
            text-align: center;
            padding: 10vh 20px 30px 20px;
        }}

        .login-logo {{
            font-size: 5rem;
            filter:
                drop-shadow(0 0 35px rgba({glow},0.65));
            margin-bottom: 10px;
        }}

        .login-title {{
            font-size: clamp(3rem,7vw,5rem);
            font-weight: 900;
            letter-spacing: -0.06em;
            background:
                linear-gradient(
                    135deg,
                    #FFFFFF,
                    var(--accent-light),
                    var(--accent)
                );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .login-subtitle {{
            color: var(--muted) !important;
            font-size: 0.8rem;
            letter-spacing: 0.25em;
            font-weight: 700;
            margin-bottom: 30px;
        }}

        /* ====================================================
           HERO
           ==================================================== */

        .hero {{
            text-align: center;
            padding: 25px 20px 50px 20px;
            animation: fadeInUp 0.7s ease;
        }}

        .hero-badge {{
            display: inline-block;
            padding: 8px 18px;
            border-radius: 999px;
            border:
                1px solid rgba({glow},0.35);
            background:
                rgba({glow},0.08);
            color: var(--accent-light) !important;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            margin-bottom: 20px;
            box-shadow:
                0 0 30px rgba({glow},0.08);
        }}

        .hero h1 {{
            font-size:
                clamp(3rem,7vw,6rem) !important;

            font-weight: 900 !important;

            line-height: 1.0;

            margin-bottom: 20px !important;

            background:
                linear-gradient(
                    135deg,
                    #FFFFFF 10%,
                    var(--accent-bright) 42%,
                    var(--accent) 75%,
                    #FFFFFF 100%
                );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;

            filter:
                drop-shadow(
                    0 8px 30px rgba({glow},0.12)
                );
        }}

        .hero-description {{
            max-width: 850px;
            margin: auto;
            font-size: 1.08rem;
            color: var(--muted) !important;
        }}

        /* ====================================================
           CARDS
           ==================================================== */

        .luxury-card {{
            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.065),
                    rgba(255,255,255,0.018)
                ) !important;

            border:
                1px solid rgba({glow},0.20) !important;

            border-radius: 22px !important;

            padding: 26px !important;

            box-shadow:
                0 22px 70px rgba(0,0,0,0.40),
                inset 0 1px 0 rgba(255,255,255,0.04);

            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);

            margin-bottom: 20px;
        }}

        .luxury-card:hover {{
            border-color:
                rgba({glow},0.48) !important;

            box-shadow:
                0 28px 80px rgba(0,0,0,0.52),
                0 0 35px rgba({glow},0.07);
        }}

        /* ====================================================
           METRIC CARDS
           ==================================================== */

        .metric-card {{
            position: relative;
            overflow: hidden;

            background:
                linear-gradient(
                    145deg,
                    #151922,
                    #0B0D12
                ) !important;

            border:
                1px solid rgba({glow},0.25) !important;

            border-radius: 20px !important;

            padding: 26px 15px !important;

            text-align: center !important;

            min-height: 125px;

            box-shadow:
                0 18px 55px rgba(0,0,0,0.35);

            transition:
                transform 0.25s ease,
                border-color 0.25s ease,
                box-shadow 0.25s ease;
        }}

        .metric-card::before {{
            content: "";
            position: absolute;
            top: -70px;
            left: 50%;
            width: 180px;
            height: 180px;
            transform: translateX(-50%);
            background:
                radial-gradient(
                    circle,
                    rgba({glow},0.12),
                    transparent 65%
                );
            pointer-events: none;
        }}

        .metric-card:hover {{
            transform: translateY(-6px);
            border-color:
                rgba({glow},0.55) !important;
            box-shadow:
                0 25px 65px rgba(0,0,0,0.48),
                0 0 30px rgba({glow},0.08);
        }}

        .metric-card h3 {{
            position: relative;
            color: var(--accent-light) !important;
            font-size: 2.25rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
        }}

        .metric-card p {{
            position: relative;
            color: #AEB7C5 !important;
            font-size: 0.74rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-top: 7px;
        }}

        /* ====================================================
           FEATURE CARDS
           ==================================================== */

        .feature-grid {{
            display: grid;
            grid-template-columns:
                repeat(auto-fit,minmax(245px,1fr));
            gap: 18px;
            margin: 25px 0;
        }}

        .feature-item {{
            background:
                linear-gradient(
                    145deg,
                    #151922,
                    #0B0D12
                );

            border:
                1px solid rgba(255,255,255,0.075);

            border-radius: 20px;

            padding: 26px;

            text-align: center;

            min-height: 205px;

            transition:
                all 0.28s ease;
        }}

        .feature-item:hover {{
            transform: translateY(-7px);
            border-color:
                rgba({glow},0.42);
            box-shadow:
                0 20px 60px rgba(0,0,0,0.42),
                0 0 25px rgba({glow},0.06);
        }}

        .feature-icon {{
            font-size: 2.7rem;
            margin-bottom: 13px;
            filter:
                drop-shadow(0 0 10px rgba({glow},0.20));
        }}

        .feature-title {{
            color: var(--accent-light) !important;
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 9px;
        }}

        .feature-desc {{
            color: #AAB2C0 !important;
            font-size: 0.88rem;
            line-height: 1.65;
        }}

        /* ====================================================
           INPUTS
           ==================================================== */

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {{
            background:
                #151922 !important;

            color:
                #FFFFFF !important;

            -webkit-text-fill-color:
                #FFFFFF !important;

            caret-color:
                var(--accent-light) !important;

            border:
                1px solid rgba({glow},0.25) !important;

            border-radius:
                12px !important;

            min-height:
                46px !important;

            font-weight:
                600 !important;
        }}

        .stTextInput input:hover,
        .stNumberInput input:hover,
        .stTextArea textarea:hover {{
            border-color:
                rgba({glow},0.48) !important;
        }}

        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus {{
            background:
                #191E29 !important;

            color:
                #FFFFFF !important;

            -webkit-text-fill-color:
                #FFFFFF !important;

            border:
                1px solid var(--accent) !important;

            box-shadow:
                0 0 0 3px rgba({glow},0.13),
                0 0 25px rgba({glow},0.06) !important;
        }}

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {{
            color:
                #727C8B !important;
            opacity:
                1 !important;
        }}

        .stNumberInput button {{
            background:
                #202631 !important;

            color:
                var(--accent-light) !important;

            border:
                1px solid rgba({glow},0.15) !important;
        }}

        .stNumberInput button:hover {{
            background:
                #2A313E !important;
            color:
                #FFFFFF !important;
        }}

        /* ====================================================
           SELECTBOX
           ==================================================== */

        div[data-baseweb="select"] > div {{
            background:
                #151922 !important;

            color:
                #FFFFFF !important;

            border:
                1px solid rgba({glow},0.25) !important;

            border-radius:
                12px !important;

            min-height:
                46px !important;
        }}

        div[data-baseweb="select"] span {{
            color:
                #FFFFFF !important;
        }}

        div[data-baseweb="popover"] {{
            background:
                #0F1218 !important;

            border:
                1px solid rgba({glow},0.20) !important;
        }}

        div[data-baseweb="menu"] {{
            background:
                #10141B !important;
        }}

        div[data-baseweb="menu"] * {{
            color:
                #FFFFFF !important;
        }}

        div[data-baseweb="option"] {{
            background:
                #10141B !important;
        }}

        div[data-baseweb="option"]:hover {{
            background:
                rgba({glow},0.15) !important;
        }}

        /* ====================================================
           BUTTONS
           ==================================================== */

        .stButton > button,
        .stFormSubmitButton > button {{
            background:
                linear-gradient(
                    135deg,
                    var(--accent),
                    var(--accent-light)
                ) !important;

            color:
                #07080C !important;

            -webkit-text-fill-color:
                #07080C !important;

            border:
                none !important;

            border-radius:
                12px !important;

            font-weight:
                850 !important;

            min-height:
                46px !important;

            box-shadow:
                0 9px 28px rgba({glow},0.17) !important;

            transition:
                all 0.22s ease !important;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            transform:
                translateY(-3px) !important;

            box-shadow:
                0 15px 40px rgba({glow},0.32) !important;

            filter:
                brightness(1.06);
        }}

        .stButton > button:active,
        .stFormSubmitButton > button:active {{
            transform:
                translateY(0) !important;
        }}

        .stDownloadButton > button {{
            background:
                linear-gradient(
                    135deg,
                    var(--accent),
                    var(--accent-light)
                ) !important;

            color:
                #07080C !important;

            -webkit-text-fill-color:
                #07080C !important;

            border:
                none !important;

            border-radius:
                12px !important;

            font-weight:
                850 !important;
        }}

        /* ====================================================
           RADIO
           ==================================================== */

        [data-testid="stRadio"] label {{
            color:
                #E9EDF3 !important;
        }}

        [data-testid="stRadio"] label p {{
            color:
                #E9EDF3 !important;
        }}

        /* ====================================================
           DATAFRAME
           ==================================================== */

        [data-testid="stDataFrame"] {{
            border:
                1px solid rgba({glow},0.22) !important;

            border-radius:
                14px !important;

            overflow:
                hidden !important;

            background:
                #10131A !important;
        }}

        [data-testid="stDataFrame"] * {{
            color:
                #F5F7FA !important;
        }}

        /* ====================================================
           EXPANDERS
           ==================================================== */

        [data-testid="stExpander"] {{
            background:
                #10141B !important;

            border:
                1px solid rgba({glow},0.20) !important;

            border-radius:
                14px !important;
        }}

        [data-testid="stExpander"] * {{
            color:
                #F5F7FA !important;
        }}

        [data-testid="stExpander"] p {{
            color:
                #B7C0CC !important;
        }}

        /* ====================================================
           ALERTS
           ==================================================== */

        [data-testid="stAlert"] {{
            background:
                #151922 !important;

            border-radius:
                12px !important;

            border:
                1px solid rgba({glow},0.20) !important;
        }}

        [data-testid="stAlert"] * {{
            color:
                #FFFFFF !important;
        }}

        /* ====================================================
           TABS
           ==================================================== */

        button[data-baseweb="tab"] {{
            color:
                #AEB7C5 !important;
            font-weight:
                700 !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color:
                var(--accent-light) !important;
        }}

        /* ====================================================
           PROGRESS
           ==================================================== */

        [data-testid="stProgressBar"] > div > div > div {{
            background:
                linear-gradient(
                    90deg,
                    var(--secondary),
                    var(--accent-light)
                ) !important;
        }}

        /* ====================================================
           CUSTOM STATUS
           ==================================================== */

        .status-box {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            border-radius: 12px;
            background:
                rgba({glow},0.07);
            border:
                1px solid rgba({glow},0.20);
            margin-bottom: 20px;
        }}

        .status-dot {{
            width: 10px;
            height: 10px;
            background:
                var(--accent-light);
            border-radius: 50%;
            box-shadow:
                0 0 15px rgba({glow},0.8);
            animation:
                pulse 2s infinite;
        }}

        .status-text {{
            color:
                var(--accent-light) !important;
            font-weight:
                700;
            font-size:
                0.85rem;
        }}

        /* ====================================================
           SECTION HEADER
           ==================================================== */

        .section-header {{
            padding:
                15px 0 20px 0;
        }}

        .section-header h2 {{
            margin-bottom:
                4px !important;
        }}

        .section-header p {{
            margin-top:
                0 !important;
        }}

        /* ====================================================
           PREDICTION RESULT
           ==================================================== */

        .prediction-result {{
            text-align:
                center;

            padding:
                35px 20px;

            background:
                radial-gradient(
                    circle at center,
                    rgba({glow},0.13),
                    transparent 60%
                );

            border:
                1px solid rgba({glow},0.28);

            border-radius:
                24px;

            box-shadow:
                0 25px 80px rgba(0,0,0,0.35);
        }}

        .prediction-species {{
            font-size:
                clamp(2rem,5vw,3.8rem);

            font-weight:
                900;

            color:
                var(--accent-light) !important;

            margin-bottom:
                10px;
        }}

        .confidence {{
            font-size:
                1.2rem;

            color:
                var(--accent-bright) !important;

            font-weight:
                750;
        }}

        /* ====================================================
           TABLE-LIKE INFO
           ==================================================== */

        .info-row {{
            display:
                flex;

            justify-content:
                space-between;

            align-items:
                center;

            padding:
                13px 0;

            border-bottom:
                1px solid rgba(255,255,255,0.06);
        }}

        .info-label {{
            color:
                #8E98A7 !important;

            font-size:
                0.85rem;
        }}

        .info-value {{
            color:
                #F5F7FA !important;

            font-weight:
                750;
        }}

        /* ====================================================
           FOOTER
           ==================================================== */

        .footer {{
            text-align:
                center;

            margin-top:
                70px;

            padding:
                30px 20px;

            border-top:
                1px solid rgba({glow},0.15);

            color:
                #707988 !important;

            font-size:
                0.8rem;
        }}

        /* ====================================================
           ANIMATIONS
           ==================================================== */

        @keyframes fadeInUp {{
            from {{
                opacity:
                    0;
                transform:
                    translateY(18px);
            }}

            to {{
                opacity:
                    1;
                transform:
                    translateY(0);
            }}
        }}

        @keyframes pulse {{
            0% {{
                box-shadow:
                    0 0 0 0 rgba({glow},0.60);
            }}

            70% {{
                box-shadow:
                    0 0 0 9px rgba({glow},0);
            }}

            100% {{
                box-shadow:
                    0 0 0 0 rgba({glow},0);
            }}
        }}

        /* ====================================================
           MOBILE
           ==================================================== */

        @media (max-width: 768px) {{

            .block-container {{
                padding:
                    1rem !important;
            }}

            .hero {{
                padding:
                    15px 8px 30px;
            }}

            .metric-card {{
                margin-bottom:
                    12px;
            }}

            .metric-card h3 {{
                font-size:
                    1.55rem !important;
            }}

            .feature-grid {{
                grid-template-columns:
                    1fr;
            }}

            .luxury-card {{
                padding:
                    18px !important;
            }}
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AUTHENTICATION
# ============================================================

# Authentication needs a small base style before full app CSS.
if not check_password():
    st.stop()


# ============================================================
# LOAD CURRENT UI
# ============================================================

load_css(
    st.session_state.font_choice,
    st.session_state.theme_choice,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_iris_data():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names,
    )

    df["species"] = iris.target

    df["species_name"] = df["species"].apply(
        lambda x: iris.target_names[x]
    )

    return (
        df,
        iris.target_names,
        iris.feature_names,
    )


df, target_names, feature_names = load_iris_data()


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_models():

    X = df[feature_names]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    random_state=42,
                ),
            ),
        ]),

        "Decision Tree": Pipeline([
            (
                "clf",
                DecisionTreeClassifier(
                    random_state=42,
                ),
            ),
        ]),

        "Random Forest": Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]),

        "K-Nearest Neighbors": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=5,
                ),
            ),
        ]),

        "Support Vector Machine": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "clf",
                SVC(
                    probability=True,
                    random_state=42,
                ),
            ),
        ]),
    }

    results = {}
    trained_pipelines = {}

    for name, pipe in models.items():

        pipe.fit(
            X_train,
            y_train,
        )

        y_pred = pipe.predict(X_test)

        results[name] = {

            "Accuracy": accuracy_score(
                y_test,
                y_pred,
            ),

            "Precision": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "F1 Score": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "Confusion Matrix": confusion_matrix(
                y_test,
                y_pred,
            ),
        }

        trained_pipelines[name] = pipe

    best_model_name = max(
        results,
        key=lambda k: results[k]["Accuracy"],
    )

    best_pipeline = trained_pipelines[
        best_model_name
    ]

    return (
        trained_pipelines,
        results,
        best_model_name,
        best_pipeline,
        X_test,
        y_test,
    )


(
    pipelines,
    model_results,
    best_model_name,
    best_pipeline,
    X_test,
    y_test,
) = train_models()


# ============================================================
# SPECIES MEANS
# ============================================================

@st.cache_data
def compute_species_means():

    return df.groupby(
        "species_name"
    )[feature_names].mean()


species_means = compute_species_means()


# ============================================================
# SESSION STATE
# ============================================================

if "best_model" not in st.session_state:
    st.session_state.best_model = best_pipeline

if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = best_model_name


# ============================================================
# SIDEBAR BRAND
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🌸</div>
        <div class="sidebar-brand-title">IrisAI</div>
        <div class="sidebar-brand-sub">
            PREMIUM ML PLATFORM
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")


# ============================================================
# LIVE THEME CONTROL
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        color:#AAB3C0;
        font-size:0.68rem;
        letter-spacing:0.16em;
        font-weight:800;
        margin-bottom:7px;">
        ✦ LUXURY COLOR SYSTEM
    </div>
    """,
    unsafe_allow_html=True,
)

theme_selection = st.sidebar.selectbox(
    "Choose visual theme",
    list(LUXURY_THEMES.keys()),
    index=list(LUXURY_THEMES.keys()).index(
        st.session_state.theme_choice
    ),
)

if theme_selection != st.session_state.theme_choice:

    st.session_state.theme_choice = theme_selection

    st.rerun()


# ============================================================
# FONT CONTROL
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        color:#AAB3C0;
        font-size:0.68rem;
        letter-spacing:0.16em;
        font-weight:800;
        margin-top:15px;
        margin-bottom:7px;">
        ✦ INTERFACE FONT
    </div>
    """,
    unsafe_allow_html=True,
)

font_selection = st.sidebar.selectbox(
    "Choose your font",
    list(FONT_OPTIONS.keys()),
    index=list(FONT_OPTIONS.keys()).index(
        st.session_state.font_choice
    ),
)

if font_selection != st.session_state.font_choice:

    st.session_state.font_choice = font_selection

    st.rerun()


st.sidebar.markdown(
    f"""
    <div style="
        margin-top:10px;
        padding:12px;
        border-radius:12px;
        background:rgba(255,255,255,0.035);
        border:1px solid rgba(255,255,255,0.07);
        text-align:center;">

        <div style="
            color:#7D8796;
            font-size:0.65rem;
            letter-spacing:0.12em;
            font-weight:800;">
            ACTIVE STYLE
        </div>

        <div style="
            color:#F4D06F;
            font-size:0.9rem;
            font-weight:800;
            margin-top:5px;">
            {font_selection}
        </div>

        <div style="
            color:#7D8796;
            font-size:0.72rem;
            margin-top:3px;">
            {theme_selection}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown("---")

nav_options = [
    "🏠 Home",
    "🤖 AI Prediction",
    "📊 Dataset Explorer",
    "📈 Data Visualization",
    "🧠 Model Performance",
    "🔬 Explainable AI",
    "📚 About Project",
]

selected_page = st.sidebar.radio(
    "📌 NAVIGATION",
    nav_options,
    index=(
        nav_options.index(
            st.session_state.page
        )
        if st.session_state.page in nav_options
        else 0
    ),
)

if selected_page != st.session_state.page:

    st.session_state.page = selected_page

    st.rerun()


st.sidebar.markdown("---")


# ============================================================
# SIDEBAR MODEL STATUS
# ============================================================

best_accuracy = model_results[
    best_model_name
]["Accuracy"]


st.sidebar.markdown(
    f"""
    <div style="
        padding:16px;
        border-radius:16px;
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.045),
                rgba(255,255,255,0.015)
            );
        border:1px solid rgba(212,175,55,0.16);">

        <div style="
            color:#7D8796;
            font-size:0.65rem;
            letter-spacing:0.12em;
            font-weight:800;">
            BEST MODEL
        </div>

        <div style="
            color:#F4D06F;
            font-weight:850;
            margin-top:5px;">
            {best_model_name}
        </div>

        <div style="
            color:#AAB3C0;
            font-size:0.78rem;
            margin-top:5px;">
            Test Accuracy:
            <strong style="color:#F4D06F;">
                {best_accuracy:.1%}
            </strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def luxury_card(content):

    st.markdown(
        f"""
        <div class="luxury-card">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate_to(page_name):

    st.session_state.page = page_name

    st.rerun()


def luxury_back_button():

    if st.button(
        "← Back to Home",
        key=f"back_home_{st.session_state.page}",
        use_container_width=False,
    ):

        navigate_to("🏠 Home")


def plotly_luxury_layout(fig, title=None):

    theme = LUXURY_THEMES[
        st.session_state.theme_choice
    ]

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#F5F7FA",
            family="Inter",
        ),
        title_font=dict(
            color=theme["accent_light"],
            size=20,
        ),
        xaxis=dict(
            color="#B8C0CB",
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.10)",
        ),
        yaxis=dict(
            color="#B8C0CB",
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.10)",
        ),
        legend=dict(
            font=dict(
                color="#F5F7FA",
            ),
        ),
        margin=dict(
            l=40,
            r=30,
            t=70 if title else 30,
            b=40,
        ),
    )

    return fig


# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == "🏠 Home":

    current_accuracy = model_results[
        best_model_name
    ]["Accuracy"]

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                ✦ ARTIFICIAL INTELLIGENCE • MACHINE LEARNING • IRIS
            </div>

            <h1>IrisAI</h1>

            <div class="hero-description">
                The Future of Flower Classification —
                AI-Powered, Real-Time & Intelligent.
                A premium machine-learning experience built
                around the classic Iris dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status-box">
            <div class="status-dot"></div>
            <div class="status-text">
                AI SYSTEM ONLINE • ALL MODELS READY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <h3>150</h3>
                <p>Total Samples</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{current_accuracy:.1%}</h3>
                <p>Best Accuracy</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
                <h3>5</h3>
                <p>ML Algorithms</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{st.session_state.prediction_count}</h3>
                <p>Predictions</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## 🚀 Why IrisAI?")

    st.markdown(
        """
        <div class="feature-grid">

            <div class="feature-item">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">
                    5 Powerful Models
                </div>
                <div class="feature-desc">
                    Logistic Regression, Decision Tree,
                    Random Forest, KNN and SVM.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">
                    Automatic Model Selection
                </div>
                <div class="feature-desc">
                    The highest-performing model on the
                    fixed test split is selected automatically.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">🔬</div>
                <div class="feature-title">
                    Explainable AI
                </div>
                <div class="feature-desc">
                    Explore which features contribute most
                    to model performance.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-title">
                    Interactive Analytics
                </div>
                <div class="feature-desc">
                    Explore distributions, relationships,
                    statistics and model metrics.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">
                    Instant Predictions
                </div>
                <div class="feature-desc">
                    Enter flower measurements and receive
                    an instant classification.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">💎</div>
                <div class="feature-title">
                    Premium Experience
                </div>
                <div class="feature-desc">
                    Luxury interface with live themes,
                    premium fonts and responsive design.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🎯 Quick Access")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🤖 AI Prediction",
            use_container_width=True,
            key="home_prediction",
        ):
            navigate_to("🤖 AI Prediction")

        if st.button(
            "📊 Dataset Explorer",
            use_container_width=True,
            key="home_dataset",
        ):
            navigate_to("📊 Dataset Explorer")

    with col2:

        if st.button(
            "📈 Data Visualization",
            use_container_width=True,
            key="home_visualization",
        ):
            navigate_to("📈 Data Visualization")

        if st.button(
            "🧠 Model Performance",
            use_container_width=True,
            key="home_models",
        ):
            navigate_to("🧠 Model Performance")

    with col3:

        if st.button(
            "🔬 Explainable AI",
            use_container_width=True,
            key="home_xai",
        ):
            navigate_to("🔬 Explainable AI")

        if st.button(
            "📚 About Project",
            use_container_width=True,
            key="home_about",
        ):
            navigate_to("📚 About Project")

    luxury_card(
        f"""
        <div style="text-align:center;">

            <div style="
                color:#7D8796;
                font-size:0.7rem;
                font-weight:800;
                letter-spacing:0.15em;">
                CURRENT INTELLIGENCE ENGINE
            </div>

            <div style="
                color:#F4D06F;
                font-size:1.6rem;
                font-weight:900;
                margin-top:8px;">
                {best_model_name}
            </div>

            <div style="
                color:#AAB3C0;
                margin-top:6px;">
                Test accuracy:
                <strong style="color:#F4D06F;">
                    {current_accuracy:.1%}
                </strong>
            </div>

        </div>
        """
    )


# ============================================================
# PAGE 2 — AI PREDICTION
# ============================================================

elif st.session_state.page == "🤖 AI Prediction":

    luxury_back_button()

    st.markdown(
        """
        <div class="section-header">

            <h1>🤖 Predict Iris Species</h1>

            <p>
                Enter the four flower measurements and let
                the selected AI model classify the species.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-box">
            <div class="status-dot"></div>
            <div class="status-text">
                ACTIVE MODEL • {best_model_name.upper()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(
        key="prediction_form"
    ):

        st.markdown(
            "### 🌸 Flower Measurements"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            sepal_len = st.number_input(
                "📏 Sepal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=5.1,
                step=0.1,
            )

        with col2:

            sepal_wid = st.number_input(
                "📐 Sepal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=3.5,
                step=0.1,
            )

        with col3:

            petal_len = st.number_input(
                "📏 Petal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=1.4,
                step=0.1,
            )

        with col4:

            petal_wid = st.number_input(
                "📐 Petal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=0.2,
                step=0.1,
            )

        submitted = st.form_submit_button(
            "🔮 Predict Species",
            use_container_width=True,
        )

    if submitted:

        try:

            input_data = pd.DataFrame(
                [[
                    sepal_len,
                    sepal_wid,
                    petal_len,
                    petal_wid,
                ]],
                columns=feature_names,
            )

            model = st.session_state.best_model

            prediction = model.predict(
                input_data
            )[0]

            probabilities = model.predict_proba(
                input_data
            )[0]

            pred_species = target_names[
                prediction
            ]

            confidence = float(
                np.max(probabilities)
            )

            st.session_state.prediction_count += 1

            prediction_record = {
                "species": pred_species,
                "confidence": confidence,
                "inputs": [
                    sepal_len,
                    sepal_wid,
                    petal_len,
                    petal_wid,
                ],
                "timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
            }

            st.session_state.last_prediction = (
                prediction_record
            )

            st.session_state.history.append(
                prediction_record
            )

            st.markdown("---")

            st.markdown(
                f"""
                <div class="prediction-result">

                    <div style="
                        font-size:3rem;
                        margin-bottom:8px;">
                        🌸
                    </div>

                    <div style="
                        color:#AAB3C0;
                        font-size:0.72rem;
                        letter-spacing:0.16em;
                        font-weight:800;">
                        PREDICTED SPECIES
                    </div>

                    <div class="prediction-species">
                        {pred_species}
                    </div>

                    <div class="confidence">
                        Confidence: {confidence:.1%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 📊 Prediction Probabilities"
            )

            prob_df = pd.DataFrame(
                {
                    "Species": target_names,
                    "Probability": probabilities,
                }
            ).sort_values(
                "Probability",
                ascending=True,
            )

            theme = LUXURY_THEMES[
                st.session_state.theme_choice
            ]

            fig = px.bar(
                prob_df,
                x="Probability",
                y="Species",
                orientation="h",
                color="Probability",
                color_continuous_scale=[
                    theme["secondary"],
                    theme["accent"],
                    theme["accent_light"],
                ],
            )

            fig.update_layout(
                coloraxis_showscale=False
            )

            fig = plotly_luxury_layout(
                fig,
                "Classification Confidence",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.success(
                f"🌸 AI classified the flower as "
                f"**{pred_species}** with "
                f"**{confidence:.1%} confidence**."
            )

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {str(e)}"
            )

    if st.session_state.last_prediction:

        st.markdown("### 📋 Last Prediction")

        last = st.session_state.last_prediction

        luxury_card(
            f"""
            <div class="info-row">
                <span class="info-label">
                    Species
                </span>
                <span class="info-value">
                    🌸 {last['species']}
                </span>
            </div>

            <div class="info-row">
                <span class="info-label">
                    Confidence
                </span>
                <span class="info-value">
                    {last['confidence']:.2%}
                </span>
            </div>

            <div class="info-row">
                <span class="info-label">
                    Timestamp
                </span>
                <span class="info-value">
                    {last['timestamp']}
                </span>
            </div>
            """
        )

    if st.session_state.history:

        with st.expander(
            "📜 Prediction History — Current Session"
        ):

            history_rows = []

            for item in st.session_state.history:

                history_rows.append(
                    {
                        "Species":
                            item["species"],
                        "Confidence":
                            f"{item['confidence']:.2%}",
                        "Sepal Length":
                            item["inputs"][0],
                        "Sepal Width":
                            item["inputs"][1],
                        "Petal Length":
                            item["inputs"][2],
                        "Petal Width":
                            item["inputs"][3],
                        "Timestamp":
                            item["timestamp"],
                    }
                )

            hist_df = pd.DataFrame(
                history_rows
            )

            st.dataframe(
                hist_df,
                use_container_width=True,
            )

            csv = hist_df.to_csv(
                index=False
            ).encode()

            st.download_button(
                "📥 Download History as CSV",
                csv,
                "prediction_history.csv",
                "text/csv",
                use_container_width=True,
            )


# ============================================================
# PAGE 3 — DATASET EXPLORER
# ============================================================

elif st.session_state.page == "📊 Dataset Explorer":

    luxury_back_button()

    st.markdown(
        """
        <div class="section-header">
            <h1>📊 Dataset Explorer</h1>
            <p>
                Explore the complete built-in Iris dataset.
                No external CSV file is required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0],
    )

    col2.metric(
        "Columns",
        df.shape[1],
    )

    col3.metric(
        "Features",
        len(feature_names),
    )

    col4.metric(
        "Missing Values",
        int(df.isnull().sum().sum()),
    )

    st.markdown("### 🔍 Dataset Search")

    search = st.text_input(
        "Filter by species name",
        placeholder="Example: setosa",
    )

    if search:

        filtered = df[
            df["species_name"].str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    else:

        filtered = df

    st.dataframe(
        filtered,
        use_container_width=True,
        height=420,
    )

    st.markdown("### 📈 Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True,
    )

    st.markdown("### 🌸 Class Distribution")

    class_counts = (
        df["species_name"]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = [
        "Species",
        "Count",
    ]

    theme = LUXURY_THEMES[
        st.session_state.theme_choice
    ]

    fig = px.bar(
        class_counts,
        x="Species",
        y="Count",
        color="Species",
        color_discrete_sequence=[
            theme["accent"],
            theme["accent_light"],
            theme["accent_bright"],
        ],
    )

    fig = plotly_luxury_layout(
        fig,
        "Samples per Species",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# PAGE 4 — DATA VISUALIZATION
# ============================================================

elif st.session_state.page == "📈 Data Visualization":

    luxury_back_button()

    st.markdown(
        """
        <div class="section-header">
            <h1>📈 Data Visualization</h1>
            <p>
                Interactive visual analytics for the Iris dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme = LUXURY_THEMES[
        st.session_state.theme_choice
    ]

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔵 Scatter Analysis",
            "📦 Feature Distribution",
            "🔥 Correlation",
            "🌸 Species Comparison",
        ]
    )

    with tab1:

        feature_x = st.selectbox(
            "X-axis Feature",
            feature_names,
            index=0,
        )

        feature_y = st.selectbox(
            "Y-axis Feature",
            feature_names,
            index=2,
        )

        fig = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            color="species_name",
            size_max=12,
            hover_data=feature_names,
            color_discrete_sequence=[
                theme["accent"],
                theme["accent_light"],
                theme["accent_bright"],
            ],
        )

        fig = plotly_luxury_layout(
            fig,
            f"{feature_x} vs {feature_y}",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab2:

        selected_feature = st.selectbox(
            "Select Feature",
            feature_names,
            key="distribution_feature",
        )

        fig = px.box(
            df,
            x="species_name",
            y=selected_feature,
            color="species_name",
            color_discrete_sequence=[
                theme["accent"],
                theme["accent_light"],
                theme["accent_bright"],
            ],
        )

        fig = plotly_luxury_layout(
            fig,
            f"{selected_feature} Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab3:

        correlation = df[
            feature_names
        ].corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[
                "#090B10",
                theme["secondary"],
                theme["accent"],
                theme["accent_light"],
            ],
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#FFFFFF"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab4:

        melted = df.melt(
            id_vars="species_name",
            value_vars=feature_names,
            var_name="Feature",
            value_name="Value",
        )

        fig = px.box(
            melted,
            x="Feature",
            y="Value",
            color="species_name",
            color_discrete_sequence=[
                theme["accent"],
                theme["accent_light"],
                theme["accent_bright"],
            ],
        )

        fig = plotly_luxury_layout(
            fig,
            "Feature Comparison by Species",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "🧠 Model Performance":

    luxury_back_button()

    st.markdown(
        """
        <div class="section-header">
            <h1>🧠 Model Performance</h1>
            <p>
                Compare all five classification algorithms
                using the same stratified test split.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    performance_rows = []

    for name, result in model_results.items():

        performance_rows.append(
            {
                "Model": name,
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1 Score": result["F1 Score"],
            }
        )

    performance_df = pd.DataFrame(
        performance_rows
    )

    display_df = performance_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]:

        display_df[column] = (
            display_df[column]
            .map(lambda x: f"{x:.2%}")
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🏆 Accuracy Comparison")

    theme = LUXURY_THEMES[
        st.session_state.theme_choice
    ]

    fig = px.bar(
        performance_df,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        color_continuous_scale=[
            theme["secondary"],
            theme["accent"],
            theme["accent_light"],
        ],
        text="Accuracy",
    )

    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )

    fig.update_layout(
        coloraxis_showscale=False,
    )

    fig = plotly_luxury_layout(
        fig,
        "Model Accuracy",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### 📊 Metric Comparison")

    melted = performance_df.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=[
            theme["accent"],
            theme["accent_light"],
            theme["accent_bright"],
            theme["secondary"],
        ],
    )

    fig.update_yaxes(
        range=[0, 1.08]
    )

    fig = plotly_luxury_layout(
        fig,
        "Complete Performance Comparison",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### 🥇 Best Model")

    best_result = model_results[
        best_model_name
    ]

    luxury_card(
        f"""
        <div style="text-align:center;">

            <div style="
                font-size:3rem;">
                🏆
            </div>

            <div style="
                color:#7D8796;
                font-size:0.72rem;
                letter-spacing:0.15em;
                font-weight:800;">
                TOP PERFORMER
            </div>

            <div style="
                color:#F4D06F;
                font-size:2rem;
                font-weight:900;
                margin-top:7px;">
                {best_model_name}
            </div>

            <div style="
                color:#AAB3C0;
                margin-top:8px;">
                Accuracy:
                <strong style="color:#F4D06F;">
                    {best_result['Accuracy']:.2%}
                </strong>
            </div>

        </div>
        """
    )

    st.markdown("### 🔲 Confusion Matrix")

    selected_model = st.selectbox(
        "Choose model",
        list(model_results.keys()),
    )

    cm = model_results[
        selected_model
    ]["Confusion Matrix"]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=target_names,
        y=target_names,
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count",
        },
        color_continuous_scale=[
            "#080A0E",
            theme["secondary"],
            theme["accent"],
            theme["accent_light"],
        ],
    )

    fig = plotly_luxury_layout(
        fig,
        f"Confusion Matrix — {selected_model}",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# PAGE 6 — EXPLAINABLE AI
# ============================================================

elif st.session_state.page == "🔬 Explainable AI":

    luxury_back_button()

    st.markdown(
        """
        <div class="section-header">
            <h1>🔬 Explainable AI</h1>
            <p>
                Understand which input features are most
                influential for model performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_xai_model = st.selectbox(
        "Select model to explain",
        list(pipelines.keys()),
        index=list(pipelines.keys()).index(
            best_model_name
        ),
    )

    selected_pipeline = pipelines[
        selected_xai_model
    ]

    with st.spinner(
        "Calculating feature importance..."
    ):

        importance_result = permutation_importance(
            selected_pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="accuracy",
        )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance":
                importance_result.importances_mean,
            "Std":
                importance_result.importances_std,
        }
    ).sort_values(
        "Importance",
        ascending=True,
    )

    st.markdown(
        "### 🎯 Permutation Feature Importance"
    )

    theme = LUXURY_THEMES[
        st.session_state.theme_choice
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h",
            error_x=dict(
                type="data",
                array=importance_df["Std"],
                visible=True,
            ),
            marker=dict(
                color=theme["accent"],
            ),
        )
    )

    fig = plotly_luxury_layout(
        fig,
        f"Feature Importance — {selected_xai_model}",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    top_feature = importance_df.iloc[-1]

    luxury_card(
        f"""
        <div style="text-align:center;">

            <div style="
                color:#7D8796;
                font-size:0.7rem;
                letter-spacing:0.15em;
                font-weight:800;">
                MOST INFLUENTIAL FEATURE
            </div>

            <div style="
                color:#F4D06F;
                font-size:2rem;
                font-weight:900;
                margin-top:8px;">
                {top_feature['Feature']}
            </div>

            <div style="
                color:#AAB3C0;
                margin-top:7px;">
                Mean permutation importance:
                <strong style="color:#F4D06F;">
                    {top_feature['Importance']:.4f}
                </strong>
            </div>

        </div>
        """
    )

    st.markdown("### 📚 What does this mean?")

    luxury_card(
        """
        <div style="
            color:#C1C9D4;
            line-height:1.8;">

            <strong style="color:#F4D06F;">
                Permutation importance
            </strong>
            measures how much model performance
            changes when one feature is randomly shuffled.

            <br><br>

            A larger positive value generally means that
            the model relies more heavily on that feature
            for its predictions on the test data.

            <br><br>

            This is a model-agnostic explanation technique,
            so it can be used with all five models in this
            application.

        </div>
        """
    )

    st.markdown("### 🌸 Species Feature Means")

    st.dataframe(
        species_means.style.format(
            "{:.2f}"
        ),
        use_container_width=True,
    )


# ============================================================
# PAGE 7 — ABOUT PROJECT
# ============================================================

elif st.session_state.page == "📚 About Project":

    luxury_back_button()

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                ✦ PROJECT INTELLIGENCE
            </div>

            <h1>About IrisAI</h1>

            <div class="hero-description">
                A complete machine-learning classification
                platform powered by the classic Iris dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🌸 Dataset")

    luxury_card(
        """
        <div class="info-row">
            <span class="info-label">
                Dataset
            </span>
            <span class="info-value">
                Iris Dataset
            </span>
        </div>

        <div class="info-row">
            <span class="info-label">
                Samples
            </span>
            <span class="info-value">
                150
            </span>
        </div>

        <div class="info-row">
            <span class="info-label">
                Features
            </span>
            <span class="info-value">
                4
            </span>
        </div>

        <div class="info-row">
            <span class="info-label">
                Classes
            </span>
            <span class="info-value">
                3
            </span>
        </div>

        <div class="info-row">
            <span class="info-label">
                Missing Values
            </span>
            <span class="info-value">
                0
            </span>
        </div>

        <div class="info-row">
            <span class="info-label">
                Data Source
            </span>
            <span class="info-value">
                scikit-learn built-in dataset
            </span>
        </div>
        """
    )

    st.markdown("## 🧠 Machine Learning Stack")

    st.markdown(
        """
        <div class="feature-grid">

            <div class="feature-item">
                <div class="feature-icon">🐍</div>
                <div class="feature-title">Python</div>
                <div class="feature-desc">
                    Core programming language.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Streamlit</div>
                <div class="feature-desc">
                    Interactive web application framework.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Pandas</div>
                <div class="feature-desc">
                    Dataset manipulation and analysis.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">🔢</div>
                <div class="feature-title">NumPy</div>
                <div class="feature-desc">
                    Numerical computation.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">scikit-learn</div>
                <div class="feature-desc">
                    Machine-learning models and metrics.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Plotly</div>
                <div class="feature-desc">
                    Interactive data visualization.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🏛️ Architecture")

    luxury_card(
        """
        <div style="
            color:#C4CCD7;
            line-height:2;">

            <strong style="color:#F4D06F;">
                01 — Data
            </strong>
            → Built-in Iris dataset

            <br>

            <strong style="color:#F4D06F;">
                02 — Preprocessing
            </strong>
            → StandardScaler where appropriate

            <br>

            <strong style="color:#F4D06F;">
                03 — Training
            </strong>
            → Five classification algorithms

            <br>

            <strong style="color:#F4D06F;">
                04 — Evaluation
            </strong>
            → Accuracy, Precision, Recall, F1 and Confusion Matrix

            <br>

            <strong style="color:#F4D06F;">
                05 — Selection
            </strong>
            → Highest test accuracy determines the best model

            <br>

            <strong style="color:#F4D06F;">
                06 — Prediction
            </strong>
            → Real-time classification from user measurements

            <br>

            <strong style="color:#F4D06F;">
                07 — Explainability
            </strong>
            → Permutation feature importance

        </div>
        """
    )

    st.markdown("## 🛡️ Platform Features")

    features = [
        "🔐 Password-protected access",
        "🌸 Built-in Iris dataset",
        "🤖 Five machine-learning algorithms",
        "🎯 Automatic best-model selection",
        "📊 Interactive visual analytics",
        "🔬 Explainable AI",
        "📜 Prediction history",
        "📥 CSV export",
        "🎨 Live luxury color themes",
        "✦ Powerful font selection",
        "📱 Responsive design",
        "⚡ Cached data and model resources",
    ]

    cols = st.columns(3)

    for index, feature in enumerate(features):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div style="
                    margin-bottom:12px;
                    padding:15px 17px;
                    border-radius:14px;
                    background:
                        rgba(255,255,255,0.035);
                    border:
                        1px solid rgba(255,255,255,0.07);
                    color:#E9EDF3;
                    font-weight:650;">
                    {feature}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div style="
            color:#F4D06F;
            font-weight:850;
            letter-spacing:0.10em;">
            IRISAI
        </div>

        <div style="margin-top:6px;">
            Premium Machine Learning Classification Platform
        </div>

        <div style="
            margin-top:6px;
            color:#596271 !important;">
            Built with Python • Streamlit • Pandas • NumPy •
            scikit-learn • Plotly
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
