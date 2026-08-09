"""
IrisAI – ULTIMATE BILLIONAIRE EDITION
The Most Powerful ML Platform Ever Built
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
    page_title="IRISAI | Billionaire Edition",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LUXURY THEMES
# ============================================================

LUXURY_THEMES = {
    "💎 Diamond Royal": {
        "accent": "#B8E1FF",
        "accent_light": "#D4EFFF",
        "accent_bright": "#FFFFFF",
        "glow": "184,225,255",
        "secondary": "#4DA8FF",
        "plot": ["#B8E1FF", "#D4EFFF", "#FFFFFF"],
        "bg": "linear-gradient(135deg,#0A0B15 0%,#141828 50%,#0A0B15 100%)",
    },
    "👑 Emperor Gold": {
        "accent": "#FFD700",
        "accent_light": "#FFE44D",
        "accent_bright": "#FFF5B8",
        "glow": "255,215,0",
        "secondary": "#D4AF37",
        "plot": ["#FFD700", "#FFE44D", "#FFF5B8"],
        "bg": "linear-gradient(135deg,#0D0800 0%,#1A1200 50%,#0D0800 100%)",
    },
    "💜 Royal Purple": {
        "accent": "#B388FF",
        "accent_light": "#D1B3FF",
        "accent_bright": "#F0E6FF",
        "glow": "179,136,255",
        "secondary": "#7C4DFF",
        "plot": ["#B388FF", "#D1B3FF", "#F0E6FF"],
        "bg": "linear-gradient(135deg,#0A0515 0%,#140A28 50%,#0A0515 100%)",
    },
    "💚 Emerald Empire": {
        "accent": "#00E676",
        "accent_light": "#69F0AE",
        "accent_bright": "#B9F6CA",
        "glow": "0,230,118",
        "secondary": "#00C853",
        "plot": ["#00E676", "#69F0AE", "#B9F6CA"],
        "bg": "linear-gradient(135deg,#000D05 0%,#001A0A 50%,#000D05 100%)",
    },
    "🔥 Crimson Dynasty": {
        "accent": "#FF6B6B",
        "accent_light": "#FF9E9E",
        "accent_bright": "#FFC8C8",
        "glow": "255,107,107",
        "secondary": "#D32F2F",
        "plot": ["#FF6B6B", "#FF9E9E", "#FFC8C8"],
        "bg": "linear-gradient(135deg,#0A0000 0%,#1A0505 50%,#0A0000 100%)",
    },
    "🌌 Galaxy Opal": {
        "accent": "#7C4DFF",
        "accent_light": "#B388FF",
        "accent_bright": "#E1D5FF",
        "glow": "124,77,255",
        "secondary": "#536DFE",
        "plot": ["#7C4DFF", "#B388FF", "#E1D5FF"],
        "bg": "linear-gradient(135deg,#050510 0%,#100A20 50%,#050510 100%)",
    },
    "🌊 Ocean Sapphire": {
        "accent": "#4FC3F7",
        "accent_light": "#81D4FA",
        "accent_bright": "#B3E5FC",
        "glow": "79,195,247",
        "secondary": "#0288D1",
        "plot": ["#4FC3F7", "#81D4FA", "#B3E5FC"],
        "bg": "linear-gradient(135deg,#000510 0%,#000D1A 50%,#000510 100%)",
    },
    "🌹 Rose Gold": {
        "accent": "#FFB6C1",
        "accent_light": "#FFD1DC",
        "accent_bright": "#FFE8ED",
        "glow": "255,182,193",
        "secondary": "#FF6B8A",
        "plot": ["#FFB6C1", "#FFD1DC", "#FFE8ED"],
        "bg": "linear-gradient(135deg,#0A0508 0%,#1A0A10 50%,#0A0508 100%)",
    },
}


# ============================================================
# FONT OPTIONS
# ============================================================

FONT_OPTIONS = {
    "Inter": "'Inter', sans-serif",
    "Poppins": "'Poppins', sans-serif",
    "Montserrat": "'Montserrat', sans-serif",
    "Playfair Display": "'Playfair Display', serif",
    "Space Grotesk": "'Space Grotesk', sans-serif",
    "Outfit": "'Outfit', sans-serif",
    "Orbitron": "'Orbitron', sans-serif",
    "Rajdhani": "'Rajdhani', sans-serif",
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
            st.session_state.pop("password_error", None)
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:15vh 20px 20px 20px;
            ">
                <div style="
                    font-size:6rem;
                    filter:drop-shadow(
                        0 0 50px rgba(255,215,0,0.6)
                    );
                ">
                    💎
                </div>

                <div style="
                    font-size:clamp(4rem,10vw,8rem);
                    font-weight:900;
                    letter-spacing:-0.06em;
                    background:linear-gradient(
                        135deg,
                        #FFFFFF,
                        #FFD700,
                        #FFD700,
                        #FFFFFF
                    );
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                ">
                    IrisAI
                </div>

                <div style="
                    color:#AAB3C0;
                    font-size:1rem;
                    letter-spacing:0.3em;
                    font-weight:700;
                    margin-bottom:30px;
                ">
                    BILLIONAIRE EDITION
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_input(
            "🔐 ENTER ACCESS CODE",
            type="password",
            on_change=password_entered,
            key="password",
        )

        if st.session_state.get("password_error"):
            st.error(
                '⚠️ Password not configured. Add password = '
                '"YourPassword" to .streamlit/secrets.toml'
            )

        return False

    elif not st.session_state["password_correct"]:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:15vh 20px 20px 20px;
            ">
                <div style="font-size:6rem;">🔒</div>

                <div style="
                    font-size:clamp(4rem,10vw,8rem);
                    font-weight:900;
                    letter-spacing:-0.06em;
                    background:linear-gradient(
                        135deg,
                        #FF6B6B,
                        #FF4444
                    );
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                ">
                    ACCESS DENIED
                </div>

                <div style="
                    color:#AAB3C0;
                    font-size:0.9rem;
                    letter-spacing:0.2em;
                    font-weight:700;
                    margin-bottom:30px;
                ">
                    AUTHENTICATION REQUIRED
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_input(
            "🔐 ENTER ACCESS CODE",
            type="password",
            on_change=password_entered,
            key="password",
        )

        st.error("❌ Incorrect password")

        return False

    return True


if not check_password():
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "font_choice": "Inter",
    "theme_choice": "👑 Emperor Gold",
    "page": "🏠 Home",
    "prediction_count": 0,
    "last_prediction": None,
    "history": [],
    "sound_enabled": True,
    "animations_enabled": True,
    "glass_effect": "Heavy",
    "display_mode": "Premium",
    "particles_enabled": True,
    "glow_effect": "High",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


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
    bg = theme["bg"]

    st.markdown(
        f"""
        <style>

        @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900
        &family=Poppins:wght@100;200;300;400;500;600;700;800;900
        &family=Montserrat:wght@100;200;300;400;500;600;700;800;900
        &family=Playfair+Display:wght@400;500;600;700;800;900
        &family=Space+Grotesk:wght@300;400;500;600;700
        &family=Outfit:wght@100;200;300;400;500;600;700;800;900
        &family=Orbitron:wght@400;500;600;700;800;900
        &family=Rajdhani:wght@300;400;500;600;700
        &display=swap'
        );

        :root {{
            --bg:#050608;
            --bg2:#080A0E;
            --panel:#101319;
            --panel2:#151922;
            --panel3:#1A1F29;

            --accent:{accent};
            --accent-light:{accent_light};
            --accent-bright:{accent_bright};
            --secondary:{secondary};

            --glow:{glow};

            --text:#F8FAFC;
            --text-soft:#E5E9EF;
            --muted:#AAB3C0;
            --muted2:#7D8796;

            --success:#43D17A;
            --danger:#FF5C65;
            --warning:#FFCA55;
            --info:#5EA7FF;
        }}

        html,
        body,
        .stApp,
        [class*="css"],
        button,
        input,
        textarea,
        select {{
            font-family:{font_family} !important;
        }}

        .stApp {{
            background:
                {bg}
                !important;
            color:var(--text) !important;
        }}

        .main {{
            background:transparent !important;
        }}

        .block-container {{
            max-width:1600px !important;
            padding-top:2rem !important;
            padding-bottom:5rem !important;
        }}

        h1,h2,h3,h4,h5,h6,
        p,span,label,li,strong,
        [data-testid="stMarkdownContainer"] {{
            color:var(--text) !important;
        }}

        p {{
            color:var(--muted) !important;
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    #0A0C14 0%,
                    #060810 50%,
                    #040508 100%
                ) !important;

            border-right:
                2px solid rgba({glow},0.15) !important;

            box-shadow:
                20px 0 80px rgba(0,0,0,0.6),
                inset -1px 0 0 rgba({glow},0.08) !important;

            backdrop-filter:blur(30px) !important;
        }}

        [data-testid="stSidebar"] * {{
            color:var(--text) !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color:
                rgba({glow},0.15) !important;
            margin:1.5rem 0 !important;
        }}

        .brand-container {{
            text-align:center;
            padding:20px 10px 15px;
            background:
                linear-gradient(
                    180deg,
                    rgba({glow},0.05),
                    transparent
                );
            border-radius:20px;
            margin-bottom:10px;
        }}

        .brand-icon {{
            font-size:3.5rem;
            filter:
                drop-shadow(
                    0 0 30px rgba({glow},0.5)
                );
            animation:float 3s ease-in-out infinite;
        }}

        @keyframes float {{
            0%,100% {{
                transform:translateY(0);
            }}
            50% {{
                transform:translateY(-8px);
            }}
        }}

        .brand-title {{
            font-size:2.2rem;
            font-weight:900;
            background:
                linear-gradient(
                    135deg,
                    #FFFFFF,
                    var(--accent-light),
                    var(--accent),
                    #FFFFFF
                );
            background-size:300% 300%;
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            animation:shimmer 3s ease-in-out infinite;
            letter-spacing:-0.04em;
        }}

        @keyframes shimmer {{
            0%,100% {{
                background-position:0% 50%;
            }}
            50% {{
                background-position:100% 50%;
            }}
        }}

        .brand-sub {{
            font-size:0.6rem;
            letter-spacing:0.25em;
            color:var(--muted2) !important;
            font-weight:800;
            margin-top:3px;
        }}

        .brand-divider {{
            height:2px;
            background:
                linear-gradient(
                    90deg,
                    transparent,
                    var(--accent),
                    transparent
                );
            margin:15px 0;
            opacity:0.3;
        }}

        .billionaire-card {{
            background:
                linear-gradient(
                    145deg,
                    rgba({glow},0.06),
                    rgba(255,255,255,0.018)
                ) !important;

            border:
                1px solid rgba({glow},0.16) !important;

            border-radius:24px !important;

            padding:30px !important;

            box-shadow:
                0 25px 70px rgba(0,0,0,0.45),
                0 0 50px rgba({glow},0.04) !important;

            backdrop-filter:blur(20px) !important;

            position:relative;
            overflow:hidden;

            transition:all 0.35s ease !important;
        }}

        .billionaire-card:hover {{
            border-color:
                rgba({glow},0.4) !important;

            transform:
                translateY(-4px) !important;

            box-shadow:
                0 35px 90px rgba(0,0,0,0.55),
                0 0 60px rgba({glow},0.08) !important;
        }}

        .metric-billionaire {{
            background:
                linear-gradient(
                    145deg,
                    rgba({glow},0.07),
                    rgba(255,255,255,0.015)
                ) !important;

            border:
                1px solid rgba({glow},0.18) !important;

            border-radius:20px !important;

            padding:28px 15px !important;

            text-align:center !important;

            min-height:130px;

            box-shadow:
                0 18px 50px rgba(0,0,0,0.4) !important;

            transition:all 0.35s ease !important;
        }}

        .metric-billionaire:hover {{
            transform:
                translateY(-7px) scale(1.02) !important;

            border-color:
                rgba({glow},0.45) !important;

            box-shadow:
                0 30px 70px rgba(0,0,0,0.55),
                0 0 40px rgba({glow},0.08) !important;
        }}

        .metric-billionaire h3 {{
            color:var(--accent-light) !important;
            font-size:2.5rem !important;
            font-weight:900 !important;
            margin:0 !important;
            text-shadow:
                0 0 30px rgba({glow},0.2);
        }}

        .metric-billionaire p {{
            color:#AEB7C5 !important;
            font-size:0.7rem !important;
            font-weight:800 !important;
            letter-spacing:0.15em;
            text-transform:uppercase;
            margin-top:8px !important;
        }}

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {{
            background:
                linear-gradient(
                    135deg,
                    var(--accent),
                    var(--accent-light)
                ) !important;

            color:#07080C !important;
            -webkit-text-fill-color:#07080C !important;

            border:none !important;

            border-radius:14px !important;

            font-weight:900 !important;

            font-size:0.95rem !important;

            min-height:50px !important;

            padding:0 1.5rem !important;

            box-shadow:
                0 10px 35px rgba({glow},0.25) !important;

            transition:all 0.3s ease !important;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stDownloadButton > button:hover {{
            transform:
                translateY(-3px) scale(1.01) !important;

            box-shadow:
                0 18px 50px rgba({glow},0.4) !important;

            filter:brightness(1.08);
        }}

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {{
            background:
                rgba(15,20,32,0.96) !important;

            color:#FFFFFF !important;

            -webkit-text-fill-color:#FFFFFF !important;

            caret-color:
                var(--accent-light) !important;

            border:
                2px solid rgba({glow},0.18) !important;

            border-radius:14px !important;

            min-height:50px !important;

            font-weight:600 !important;

            padding:0 18px !important;
        }}

        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus {{
            border-color:
                var(--accent) !important;

            box-shadow:
                0 0 0 4px rgba({glow},0.14),
                0 0 30px rgba({glow},0.06) !important;
        }}

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {{
            color:#778292 !important;
            -webkit-text-fill-color:#778292 !important;
        }}

        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {{
            background:
                #111722 !important;

            color:#FFFFFF !important;

            border:
                2px solid rgba({glow},0.18) !important;

            border-radius:14px !important;
        }}

        .stSelectbox div[data-baseweb="select"] span,
        .stMultiSelect div[data-baseweb="select"] span {{
            color:#FFFFFF !important;
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="menu"] {{
            background:#10151F !important;
            border:1px solid rgba({glow},0.2) !important;
        }}

        div[data-baseweb="menu"] * {{
            color:#FFFFFF !important;
        }}

        [data-testid="stExpander"] {{
            background:
                rgba(15,19,29,0.95) !important;

            border:
                1px solid rgba({glow},0.18) !important;

            border-radius:16px !important;
        }}

        [data-testid="stExpander"] * {{
            color:#F5F7FA !important;
        }}

        [data-testid="stAlert"] {{
            background:
                rgba(18,23,34,0.98) !important;

            border-radius:14px !important;
        }}

        [data-testid="stAlert"] * {{
            color:#FFFFFF !important;
        }}

        .hero-billionaire {{
            text-align:center;
            padding:30px 20px 50px;
            animation:fadeInUp 0.8s ease;
        }}

        .hero-badge {{
            display:inline-block;
            padding:10px 25px;
            border-radius:999px;
            border:
                1px solid rgba({glow},0.35);

            background:
                rgba({glow},0.06);

            color:
                var(--accent-light) !important;

            font-size:0.7rem;
            font-weight:800;
            letter-spacing:0.15em;
            margin-bottom:25px;

            box-shadow:
                0 0 40px rgba({glow},0.06);
        }}

        .hero-billionaire h1 {{
            font-size:
                clamp(3.5rem,10vw,8rem) !important;

            font-weight:900 !important;

            line-height:1 !important;

            margin-bottom:25px !important;

            background:
                linear-gradient(
                    135deg,
                    #FFFFFF 10%,
                    var(--accent-bright) 40%,
                    var(--accent) 70%,
                    #FFFFFF 100%
                );

            background-size:300% 300%;

            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;

            animation:
                shimmer 4s ease-in-out infinite;
        }}

        .hero-description {{
            max-width:850px;
            margin:auto;
            font-size:1.1rem;
            color:var(--muted) !important;
            line-height:1.8;
        }}

        .status-billionaire {{
            display:flex;
            align-items:center;
            gap:12px;
            padding:14px 20px;
            border-radius:16px;

            background:
                rgba({glow},0.04);

            border:
                1px solid rgba({glow},0.14);

            margin-bottom:20px;
        }}

        .status-dot {{
            width:12px;
            height:12px;

            background:
                var(--accent-light);

            border-radius:50%;

            box-shadow:
                0 0 20px rgba({glow},0.5);

            animation:
                pulse 2s infinite;
        }}

        .status-text {{
            color:
                var(--accent-light) !important;

            font-weight:800;
            font-size:0.8rem;
        }}

        .prediction-billionaire {{
            text-align:center;

            padding:45px 30px;

            background:
                radial-gradient(
                    circle at center,
                    rgba({glow},0.1),
                    transparent 70%
                );

            border:
                2px solid rgba({glow},0.22);

            border-radius:30px;

            box-shadow:
                0 30px 100px rgba(0,0,0,0.45),
                0 0 60px rgba({glow},0.05);
        }}

        .prediction-species {{
            font-size:
                clamp(2.5rem,6vw,4.5rem) !important;

            font-weight:900 !important;

            color:
                var(--accent-light) !important;

            margin-bottom:10px !important;
        }}

        .prediction-confidence {{
            font-size:1.3rem !important;
            color:var(--accent-bright) !important;
            font-weight:800 !important;
        }}

        .feature-grid-billionaire {{
            display:grid;

            grid-template-columns:
                repeat(auto-fit,minmax(250px,1fr));

            gap:20px;

            margin:30px 0;
        }}

        .feature-item-billionaire {{
            background:
                linear-gradient(
                    145deg,
                    rgba({glow},0.05),
                    rgba(255,255,255,0.012)
                );

            border:
                1px solid rgba({glow},0.1);

            border-radius:22px;

            padding:30px 20px;

            text-align:center;

            min-height:210px;

            transition:
                all 0.35s ease;
        }}

        .feature-item-billionaire:hover {{
            transform:
                translateY(-7px);

            border-color:
                rgba({glow},0.3);

            box-shadow:
                0 25px 60px rgba(0,0,0,0.45),
                0 0 30px rgba({glow},0.05);
        }}

        .feature-icon-billionaire {{
            font-size:3rem;
            margin-bottom:15px;
        }}

        .feature-title-billionaire {{
            color:
                var(--accent-light) !important;

            font-size:1.1rem;

            font-weight:800;

            margin-bottom:10px;
        }}

        .feature-desc-billionaire {{
            color:#AAB2C0 !important;

            font-size:0.9rem;

            line-height:1.7;
        }}

        .footer-billionaire {{
            text-align:center;

            margin-top:80px;

            padding:40px 20px;

            border-top:
                1px solid rgba({glow},0.1);

            color:#697382 !important;

            font-size:0.8rem;
        }}

        .footer-billionaire .brand {{
            color:
                var(--accent-light) !important;

            font-weight:900;

            font-size:1rem;

            letter-spacing:0.1em;
        }}

        .section-title {{
            font-size:2rem !important;
            font-weight:900 !important;

            color:
                var(--accent-light) !important;

            margin-top:25px !important;
        }}

        .small-label {{
            color:
                var(--muted) !important;

            font-size:0.7rem;

            font-weight:800;

            letter-spacing:0.14em;

            text-transform:uppercase;
        }}

        .history-item {{
            background:
                rgba({glow},0.035);

            border:
                1px solid rgba({glow},0.1);

            border-radius:15px;

            padding:15px 18px;

            margin-bottom:10px;
        }}

        .history-item strong {{
            color:
                var(--accent-light) !important;
        }}

        ::-webkit-scrollbar {{
            width:6px;
            height:6px;
        }}

        ::-webkit-scrollbar-track {{
            background:rgba(255,255,255,0.03);
        }}

        ::-webkit-scrollbar-thumb {{
            background:
                linear-gradient(
                    180deg,
                    var(--accent),
                    var(--secondary)
                );

            border-radius:10px;
        }}

        @keyframes fadeInUp {{
            from {{
                opacity:0;
                transform:translateY(30px);
            }}

            to {{
                opacity:1;
                transform:translateY(0);
            }}
        }}

        @keyframes pulse {{
            0% {{
                box-shadow:
                    0 0 0 0 rgba({glow},0.5);
            }}

            70% {{
                box-shadow:
                    0 0 0 15px rgba({glow},0);
            }}

            100% {{
                box-shadow:
                    0 0 0 0 rgba({glow},0);
            }}
        }}

        @media (max-width:768px) {{

            .block-container {{
                padding:1rem !important;
            }}

            .metric-billionaire {{
                padding:18px 10px !important;
                min-height:100px;
            }}

            .metric-billionaire h3 {{
                font-size:1.8rem !important;
            }}

            .feature-grid-billionaire {{
                grid-template-columns:1fr;
            }}

            .billionaire-card {{
                padding:20px !important;
            }}

            .hero-billionaire {{
                padding:15px 10px 30px !important;
            }}

            .hero-billionaire h1 {{
                font-size:
                    clamp(3rem,15vw,5rem) !important;
            }}
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


load_css(
    st.session_state.font_choice,
    st.session_state.theme_choice,
)


# ============================================================
# LOAD IRIS DATA
# ============================================================

@st.cache_data
def load_iris_data():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names,
    )

    df["species"] = iris.target

    df["species_name"] = df["species"].map(
        dict(enumerate(iris.target_names))
    )

    return (
        df,
        iris.target_names,
        iris.feature_names,
    )


df, target_names, feature_names = load_iris_data()


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    X = df[feature_names]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
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
                    max_iter=500,
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

        pipe.fit(X_train, y_train)

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
        key=lambda x: results[x]["Accuracy"],
    )

    return (
        trained_pipelines,
        results,
        best_model_name,
        trained_pipelines[best_model_name],
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
    return df.groupby("species_name")[feature_names].mean()


species_means = compute_species_means()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand-container">
        <div class="brand-icon">💎</div>
        <div class="brand-title">IrisAI</div>
        <div class="brand-sub">
            ULTIMATE BILLIONAIRE EDITION
        </div>
        <div class="brand-divider"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# THEME SELECTOR
# ============================================================

st.sidebar.markdown(
    '<div class="small-label">✦ Luxury Theme</div>',
    unsafe_allow_html=True,
)

theme_selection = st.sidebar.selectbox(
    "Choose luxury theme",
    list(LUXURY_THEMES.keys()),
    index=list(LUXURY_THEMES.keys()).index(
        st.session_state.theme_choice
    ),
    label_visibility="collapsed",
)

if theme_selection != st.session_state.theme_choice:

    st.session_state.theme_choice = theme_selection

    st.rerun()


# ============================================================
# FONT SELECTOR
# ============================================================

st.sidebar.markdown(
    '<div class="small-label">✦ Interface Font</div>',
    unsafe_allow_html=True,
)

font_selection = st.sidebar.selectbox(
    "Choose font",
    list(FONT_OPTIONS.keys()),
    index=list(FONT_OPTIONS.keys()).index(
        st.session_state.font_choice
    ),
    label_visibility="collapsed",
)

if font_selection != st.session_state.font_choice:

    st.session_state.font_choice = font_selection

    st.rerun()


st.sidebar.markdown(
    f"""
    <div class="billionaire-card" style="padding:15px!important;">
        <div class="small-label">ACTIVE FONT</div>
        <div style="
            color:var(--accent-light);
            font-size:1.05rem;
            font-weight:900;
            margin-top:6px;
        ">
            {font_selection}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR ADVANCED CONTROLS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    '<div class="small-label">✦ Experience Controls</div>',
    unsafe_allow_html=True,
)

st.session_state.animations_enabled = st.sidebar.toggle(
    "✨ Animations",
    value=st.session_state.animations_enabled,
)

st.session_state.particles_enabled = st.sidebar.toggle(
    "✦ Premium Effects",
    value=st.session_state.particles_enabled,
)

st.session_state.sound_enabled = st.sidebar.toggle(
    "🔊 Sound",
    value=st.session_state.sound_enabled,
)


# ============================================================
# NAVIGATION
# ============================================================

nav_options = [
    "🏠 Home",
    "🤖 AI Prediction",
    "📊 Dataset Explorer",
    "📈 Data Visualization",
    "🧠 Model Performance",
    "🔬 Explainable AI",
    "📜 Prediction History",
    "📚 About Project",
]

selected_page = st.sidebar.radio(
    "📌 NAVIGATION",
    nav_options,
    index=(
        nav_options.index(st.session_state.page)
        if st.session_state.page in nav_options
        else 0
    ),
)

st.session_state.page = selected_page


# ============================================================
# SIDEBAR STATUS
# ============================================================

best_accuracy = model_results[
    best_model_name
]["Accuracy"]

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <div class="status-billionaire">
        <div class="status-dot"></div>
        <div>
            <div class="status-text">
                SYSTEM ONLINE
            </div>
            <div style="
                color:#8D98A8;
                font-size:0.7rem;
                margin-top:3px;
            ">
                Best: {best_model_name}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def navigate_to(page_name):

    st.session_state.page = page_name
    st.rerun()


def luxury_card(content):

    st.markdown(
        f"""
        <div class="billionaire-card">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(value, label):

    st.markdown(
        f"""
        <div class="metric-billionaire">
            <h3>{value}</h3>
            <p>{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def back_home():

    if st.button("← Back to Home"):

        navigate_to("🏠 Home")


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "🏠 Home":

    st.markdown(
        """
        <div class="hero-billionaire">

            <div class="hero-badge">
                ✦ NEXT-GENERATION MACHINE LEARNING
            </div>

            <h1>IrisAI</h1>

            <div class="hero-description">
                The ultimate luxury machine-learning platform
                for intelligent Iris flower classification,
                interactive analytics and explainable AI.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("150", "Total Samples")

    with c2:
        metric_card(
            f"{best_accuracy:.1%}",
            "Best Accuracy",
        )

    with c3:
        metric_card("5", "ML Algorithms")

    with c4:
        metric_card(
            str(st.session_state.prediction_count),
            "Predictions",
        )

    st.markdown(
        '<div class="section-title">🚀 Why IrisAI?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid-billionaire">

            <div class="feature-item-billionaire">
                <div class="feature-icon-billionaire">🧠</div>
                <div class="feature-title-billionaire">
                    5 Powerful Models
                </div>
                <div class="feature-desc-billionaire">
                    Logistic Regression, Decision Tree,
                    Random Forest, KNN and SVM.
                </div>
            </div>

            <div class="feature-item-billionaire">
                <div class="feature-icon-billionaire">🎯</div>
                <div class="feature-title-billionaire">
                    Automatic Selection
                </div>
                <div class="feature-desc-billionaire">
                    The highest-performing model is
                    automatically identified.
                </div>
            </div>

            <div class="feature-item-billionaire">
                <div class="feature-icon-billionaire">🔬</div>
                <div class="feature-title-billionaire">
                    Explainable AI
                </div>
                <div class="feature-desc-billionaire">
                    Understand feature importance
                    and model behaviour.
                </div>
            </div>

            <div class="feature-item-billionaire">
                <div class="feature-icon-billionaire">📊</div>
                <div class="feature-title-billionaire">
                    Interactive Analytics
                </div>
                <div class="feature-desc-billionaire">
                    Explore the Iris dataset using
                    premium interactive charts.
                </div>
            </div>

            <div class="feature-item-billionaire">
                <div class="feature-icon-billionaire">⚡</div>
                <div class="feature-title-billionaire">
                    Instant Predictions
                </div>
                <div class="feature-desc-billionaire">
                    Enter flower measurements and
                    receive a real-time classification.
                </div>
            </div>

            <div class="feature-item-billionaire">
                <div class="feature-icon-billionaire">💎</div>
                <div class="feature-title-billionaire">
                    Luxury Experience
                </div>
                <div class="feature-desc-billionaire">
                    A premium billionaire-tech interface
                    with live themes and fonts.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">🎯 Quick Access</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        if st.button(
            "🤖 AI Prediction",
            use_container_width=True,
        ):
            navigate_to("🤖 AI Prediction")

    with q2:
        if st.button(
            "📊 Dataset Explorer",
            use_container_width=True,
        ):
            navigate_to("📊 Dataset Explorer")

    with q3:
        if st.button(
            "🧠 Model Performance",
            use_container_width=True,
        ):
            navigate_to("🧠 Model Performance")


# ============================================================
# AI PREDICTION
# ============================================================

elif st.session_state.page == "🤖 AI Prediction":

    back_home()

    st.markdown(
        """
        <div class="hero-billionaire">

            <div class="hero-badge">
                ✦ ARTIFICIAL INTELLIGENCE ENGINE
            </div>

            <h1>AI Prediction</h1>

            <div class="hero-description">
                Enter the four Iris measurements and let
                the selected machine-learning model
                classify the flower.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.25])

    with left:

        luxury_card(
            """
            <div class="small-label">
                FLOWER MEASUREMENTS
            </div>
            <h2 style="color:var(--accent-light)!important;">
                Input Features
            </h2>
            """
        )

        model_choice = st.selectbox(
            "Prediction Model",
            list(pipelines.keys()),
        )

        sepal_length = st.number_input(
            "🌿 Sepal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.8,
            step=0.1,
        )

        sepal_width = st.number_input(
            "🌿 Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
        )

        petal_length = st.number_input(
            "🌸 Petal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=4.3,
            step=0.1,
        )

        petal_width = st.number_input(
            "🌸 Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.3,
            step=0.1,
        )

        predict_button = st.button(
            "🚀 GENERATE AI PREDICTION",
            use_container_width=True,
        )

    with right:

        if predict_button:

            input_data = pd.DataFrame(
                [[
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width,
                ]],
                columns=feature_names,
            )

            model = pipelines[model_choice]

            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

            else:

                probabilities = np.zeros(
                    len(target_names)
                )

            species = target_names[prediction]

            confidence = (
                float(np.max(probabilities))
                if len(probabilities)
                else 0
            )

            st.session_state.prediction_count += 1

            st.session_state.last_prediction = species

            st.session_state.history.append(
                {
                    "Time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Model": model_choice,
                    "Species": species,
                    "Confidence": confidence,
                    "Sepal Length": sepal_length,
                    "Sepal Width": sepal_width,
                    "Petal Length": petal_length,
                    "Petal Width": petal_width,
                }
            )

            st.markdown(
                f"""
                <div class="prediction-billionaire">

                    <div class="small-label">
                        AI CLASSIFICATION RESULT
                    </div>

                    <div class="prediction-species">
                        {species.title()}
                    </div>

                    <div class="prediction-confidence">
                        Confidence: {confidence:.2%}
                    </div>

                    <div style="
                        color:#AAB3C0;
                        margin-top:15px;
                    ">
                        Model: {model_choice}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            if len(probabilities):

                prob_df = pd.DataFrame(
                    {
                        "Species": [
                            x.title()
                            for x in target_names
                        ],
                        "Probability": probabilities,
                    }
                )

                fig = px.bar(
                    prob_df,
                    x="Species",
                    y="Probability",
                    text_auto=".1%",
                    title="Prediction Probability",
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        else:

            st.markdown(
                """
                <div class="prediction-billionaire">

                    <div style="font-size:4rem;">
                        🔮
                    </div>

                    <h2 style="
                        color:var(--accent-light)!important;
                    ">
                        Awaiting Prediction
                    </h2>

                    <p>
                        Enter the measurements and launch
                        the AI engine.
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif st.session_state.page == "📊 Dataset Explorer":

    back_home()

    st.markdown(
        """
        <div class="hero-billionaire">
            <div class="hero-badge">
                ✦ DATA INTELLIGENCE
            </div>
            <h1>Dataset Explorer</h1>
            <div class="hero-description">
                Explore the classic Iris dataset
                used by the machine-learning engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("150", "Rows")

    with c2:
        metric_card("4", "Features")

    with c3:
        metric_card("3", "Classes")

    with c4:
        metric_card(
            f"{df.isna().sum().sum()}",
            "Missing Values",
        )

    st.markdown(
        '<div class="section-title">📋 Dataset</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=500,
    )

    st.markdown(
        '<div class="section-title">📐 Statistics</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df[feature_names].describe().T,
        use_container_width=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ DOWNLOAD IRIS DATASET",
        data=csv,
        file_name="iris_dataset.csv",
        mime="text/csv",
    )


# ============================================================
# VISUALIZATION
# ============================================================

elif st.session_state.page == "📈 Data Visualization":

    back_home()

    st.markdown(
        """
        <div class="hero-billionaire">
            <div class="hero-badge">
                ✦ INTERACTIVE ANALYTICS
            </div>
            <h1>Data Visualization</h1>
            <div class="hero-description">
                Discover relationships and patterns
                hidden inside the Iris dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🌸 Scatter",
            "📊 Distribution",
            "📦 Box Plot",
            "🔥 Correlation",
        ]
    )

    with tab1:

        x_feature = st.selectbox(
            "X Feature",
            feature_names,
            index=0,
        )

        y_feature = st.selectbox(
            "Y Feature",
            feature_names,
            index=2,
        )

        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color="species_name",
            size="petal width (cm)",
            hover_data=feature_names,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab2:

        feature = st.selectbox(
            "Select Feature",
            feature_names,
            key="distribution_feature",
        )

        fig = px.histogram(
            df,
            x=feature,
            color="species_name",
            marginal="box",
            barmode="overlay",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab3:

        feature = st.selectbox(
            "Select Feature",
            feature_names,
            key="box_feature",
        )

        fig = px.box(
            df,
            x="species_name",
            y=feature,
            color="species_name",
            points="all",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with tab4:

        corr = df[feature_names].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "🧠 Model Performance":

    back_home()

    st.markdown(
        """
        <div class="hero-billionaire">
            <div class="hero-badge">
                ✦ MODEL INTELLIGENCE
            </div>
            <h1>Model Performance</h1>
            <div class="hero-description">
                Compare five machine-learning algorithms
                using the same test split.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []

    for name, result in model_results.items():

        rows.append(
            {
                "Model": name,
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1 Score": result["F1 Score"],
            }
        )

    performance_df = pd.DataFrame(rows)

    st.dataframe(
        performance_df.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    fig = px.bar(
        performance_df,
        x="Model",
        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ],
        barmode="group",
        title="Model Comparison",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".0%",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div class="prediction-billionaire">

            <div class="small-label">
                CURRENT TOP PERFORMER
            </div>

            <div class="prediction-species">
                {best_model_name}
            </div>

            <div class="prediction-confidence">
                Accuracy: {best_accuracy:.2%}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_model = st.selectbox(
        "Confusion Matrix Model",
        list(model_results.keys()),
    )

    matrix = model_results[
        selected_model
    ]["Confusion Matrix"]

    fig = px.imshow(
        matrix,
        text_auto=True,
        x=[x.title() for x in target_names],
        y=[x.title() for x in target_names],
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count",
        },
        title=f"Confusion Matrix — {selected_model}",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif st.session_state.page == "🔬 Explainable AI":

    back_home()

    st.markdown(
        """
        <div class="hero-billionaire">
            <div class="hero-badge">
                ✦ EXPLAINABLE ARTIFICIAL INTELLIGENCE
            </div>
            <h1>Explainable AI</h1>
            <div class="hero-description">
                Understand which features influence
                model predictions most strongly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explain_model = st.selectbox(
        "Select Model",
        list(pipelines.keys()),
    )

    model = pipelines[explain_model]

    with st.spinner("Calculating feature importance..."):

        importance = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
        )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance.importances_mean,
            "Std": importance.importances_std,
        }
    ).sort_values(
        "Importance",
        ascending=True,
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        error_x="Std",
        title=f"Permutation Feature Importance — {explain_model}",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">🌸 Species Feature Means</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        species_means.style.format("{:.2f}"),
        use_container_width=True,
    )


# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "📜 Prediction History":

    back_home()

    st.markdown(
        """
        <div class="hero-billionaire">
            <div class="hero-badge">
                ✦ AI ACTIVITY LOG
            </div>
            <h1>Prediction History</h1>
            <div class="hero-description">
                Review previous predictions generated
                during this session.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.history:

        st.info(
            "No predictions have been generated yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            height=500,
        )

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ DOWNLOAD HISTORY",
            data=csv,
            file_name="iris_prediction_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ CLEAR HISTORY"):

            st.session_state.history = []

            st.rerun()


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "📚 About Project":

    back_home()

    st.markdown(
        """
        <div class="hero-billionaire">
            <div class="hero-badge">
                ✦ PROJECT INTELLIGENCE
            </div>
            <h1>About IrisAI</h1>
            <div class="hero-description">
                A premium machine-learning demonstration
                built around the classic Iris dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    luxury_card(
        """
        <h2 style="color:var(--accent-light)!important;">
            🌸 The Iris Dataset
        </h2>

        <p>
            The Iris dataset contains 150 samples of
            three Iris species. Each sample contains
            four numerical measurements:
            sepal length, sepal width, petal length
            and petal width.
        </p>
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        luxury_card(
            """
            <h3 style="color:var(--accent-light)!important;">
                🤖 Machine Learning
            </h3>

            <p>
                IrisAI trains and evaluates five
                classification algorithms:
            </p>

            <p>
                • Logistic Regression<br>
                • Decision Tree<br>
                • Random Forest<br>
                • K-Nearest Neighbors<br>
                • Support Vector Machine
            </p>
            """
        )

    with c2:

        luxury_card(
            """
            <h3 style="color:var(--accent-light)!important;">
                💎 Technology Stack
            </h3>

            <p>
                • Python<br>
                • Streamlit<br>
                • Pandas<br>
                • NumPy<br>
                • Scikit-learn<br>
                • Plotly
            </p>
            """
        )

    luxury_card(
        f"""
        <h3 style="color:var(--accent-light)!important;">
            🏆 Current Best Model
        </h3>

        <p>
            The automatically selected highest-performing
            model is:
        </p>

        <h2 style="color:var(--accent-light)!important;">
            {best_model_name}
        </h2>

        <p>
            Test accuracy:
            <strong style="color:var(--accent-light)!important;">
                {best_accuracy:.2%}
            </strong>
        </p>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-billionaire">

        <div class="brand">
            💎 IRISAI
        </div>

        <div style="margin-top:8px;">
            ULTIMATE BILLIONAIRE EDITION
        </div>

        <div style="margin-top:8px;">
            Premium Machine Learning • Explainable AI •
            Interactive Analytics
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)
