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
import time
import json
import base64

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
# LUXURY BILLIONAIRE THEMES
# ============================================================
LUXURY_THEMES = {
    "💎 Diamond Royal": {
        "accent": "#B8E1FF",
        "accent_light": "#D4EFFF",
        "accent_bright": "#FFFFFF",
        "glow": "184,225,255",
        "secondary": "#4DA8FF",
        "plot": ["#B8E1FF", "#D4EFFF", "#FFFFFF"],
        "bg": "linear-gradient(135deg, #0A0B15 0%, #141828 50%, #0A0B15 100%)",
    },
    "👑 Emperor Gold": {
        "accent": "#FFD700",
        "accent_light": "#FFE44D",
        "accent_bright": "#FFF5B8",
        "glow": "255,215,0",
        "secondary": "#D4AF37",
        "plot": ["#FFD700", "#FFE44D", "#FFF5B8"],
        "bg": "linear-gradient(135deg, #0D0800 0%, #1A1200 50%, #0D0800 100%)",
    },
    "💜 Royal Purple": {
        "accent": "#B388FF",
        "accent_light": "#D1B3FF",
        "accent_bright": "#F0E6FF",
        "glow": "179,136,255",
        "secondary": "#7C4DFF",
        "plot": ["#B388FF", "#D1B3FF", "#F0E6FF"],
        "bg": "linear-gradient(135deg, #0A0515 0%, #140A28 50%, #0A0515 100%)",
    },
    "💚 Emerald Empire": {
        "accent": "#00E676",
        "accent_light": "#69F0AE",
        "accent_bright": "#B9F6CA",
        "glow": "0,230,118",
        "secondary": "#00C853",
        "plot": ["#00E676", "#69F0AE", "#B9F6CA"],
        "bg": "linear-gradient(135deg, #000D05 0%, #001A0A 50%, #000D05 100%)",
    },
    "🔥 Crimson Dynasty": {
        "accent": "#FF6B6B",
        "accent_light": "#FF9E9E",
        "accent_bright": "#FFC8C8",
        "glow": "255,107,107",
        "secondary": "#D32F2F",
        "plot": ["#FF6B6B", "#FF9E9E", "#FFC8C8"],
        "bg": "linear-gradient(135deg, #0A0000 0%, #1A0505 50%, #0A0000 100%)",
    },
    "🌌 Galaxy Opal": {
        "accent": "#7C4DFF",
        "accent_light": "#B388FF",
        "accent_bright": "#E1D5FF",
        "glow": "124,77,255",
        "secondary": "#536DFE",
        "plot": ["#7C4DFF", "#B388FF", "#E1D5FF"],
        "bg": "linear-gradient(135deg, #050510 0%, #100A20 50%, #050510 100%)",
    },
    "🌊 Ocean Sapphire": {
        "accent": "#4FC3F7",
        "accent_light": "#81D4FA",
        "accent_bright": "#B3E5FC",
        "glow": "79,195,247",
        "secondary": "#0288D1",
        "plot": ["#4FC3F7", "#81D4FA", "#B3E5FC"],
        "bg": "linear-gradient(135deg, #000510 0%, #000D1A 50%, #000510 100%)",
    },
    "🌹 Rose Gold": {
        "accent": "#FFB6C1",
        "accent_light": "#FFD1DC",
        "accent_bright": "#FFE8ED",
        "glow": "255,182,193",
        "secondary": "#FF6B8A",
        "plot": ["#FFB6C1", "#FFD1DC", "#FFE8ED"],
        "bg": "linear-gradient(135deg, #0A0508 0%, #1A0A10 50%, #0A0508 100%)",
    }
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
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:15vh 20px 20px 20px;">
            <div style="font-size:6rem;filter:drop-shadow(0 0 50px rgba(255,215,0,0.6));">💎</div>
            <div style="font-size:clamp(4rem,10vw,8rem);font-weight:900;letter-spacing:-0.06em;background:linear-gradient(135deg,#FFFFFF,#FFD700,#FFD700,#FFFFFF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">IrisAI</div>
            <div style="color:#AAB3C0;font-size:1rem;letter-spacing:0.3em;font-weight:700;margin-bottom:30px;">BILLIONAIRE EDITION</div>
        </div>
        """, unsafe_allow_html=True)
        st.text_input("🔐 ENTER ACCESS CODE", type="password", on_change=password_entered, key="password")
        if st.session_state.get("password_error"):
            st.error("⚠️ Password not configured. Add password = \"YourPassword\" to .streamlit/secrets.toml")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("""
        <div style="text-align:center;padding:15vh 20px 20px 20px;">
            <div style="font-size:6rem;">🔒</div>
            <div style="font-size:clamp(4rem,10vw,8rem);font-weight:900;letter-spacing:-0.06em;background:linear-gradient(135deg,#FF6B6B,#FF4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">ACCESS DENIED</div>
            <div style="color:#AAB3C0;font-size:0.9rem;letter-spacing:0.2em;font-weight:700;margin-bottom:30px;">AUTHENTICATION REQUIRED</div>
        </div>
        """, unsafe_allow_html=True)
        st.text_input("🔐 ENTER ACCESS CODE", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect password")
        return False
    return True

if not check_password():
    st.stop()

# ============================================================
# SESSION STATE - ULTIMATE CONTROLS
# ============================================================
if "font_choice" not in st.session_state:
    st.session_state.font_choice = "Inter"
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "👑 Emperor Gold"
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "history" not in st.session_state:
    st.session_state.history = []
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False
if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = True
if "animations_enabled" not in st.session_state:
    st.session_state.animations_enabled = True
if "glass_effect" not in st.session_state:
    st.session_state.glass_effect = "Heavy"
if "display_mode" not in st.session_state:
    st.session_state.display_mode = "Premium"
if "particles_enabled" not in st.session_state:
    st.session_state.particles_enabled = True
if "glow_effect" not in st.session_state:
    st.session_state.glow_effect = "High"

# ============================================================
# ULTIMATE CSS - BILLIONAIRE EDITION
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

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&family=Poppins:wght@100;200;300;400;500;600;700;800;900&family=Montserrat:wght@100;200;300;400;500;600;700;800;900&family=Playfair+Display:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=Outfit:wght@100;200;300;400;500;600;700;800;900&family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

    /* ================================================
       GLOBAL RESET - BILLIONAIRE VIBES
       ================================================ */
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

    html, body, .stApp, [class*="css"], button, input, textarea, select {{
        font-family: {font_family} !important;
    }}

    .stApp {{
        background: {bg} !important;
        color: var(--text) !important;
    }}

    .main {{
        background: transparent !important;
    }}

    .block-container {{
        max-width: 1600px !important;
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }}

    /* ================================================
       BILLIONAIRE GLASS EFFECT
       ================================================ */
    .glass-heavy {{
        background: rgba(255,255,255,0.04) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba({glow},0.15) !important;
        box-shadow: 0 25px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba({glow},0.08) !important;
    }}

    .glass-light {{
        background: rgba(255,255,255,0.02) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba({glow},0.08) !important;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3) !important;
    }}

    .glass-premium {{
        background: linear-gradient(145deg, rgba({glow},0.06), rgba(255,255,255,0.02)) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba({glow},0.12) !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 40px rgba({glow},0.03) !important;
    }}

    /* ================================================
       SIDEBAR - ULTRA PREMIUM
       ================================================ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0A0C14 0%, #060810 50%, #040508 100%) !important;
        border-right: 2px solid rgba({glow},0.15) !important;
        box-shadow: 20px 0 80px rgba(0,0,0,0.6), inset -1px 0 0 rgba({glow},0.08) !important;
        backdrop-filter: blur(30px) !important;
        -webkit-backdrop-filter: blur(30px) !important;
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text) !important;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: rgba({glow},0.15) !important;
        margin: 1.5rem 0 !important;
    }}

    /* ================================================
       BILLIONAIRE BRAND
       ================================================ */
    .brand-container {{
        text-align: center;
        padding: 20px 10px 15px 10px;
        background: linear-gradient(180deg, rgba({glow},0.05), transparent);
        border-radius: 20px;
        margin-bottom: 10px;
    }}

    .brand-icon {{
        font-size: 3.5rem;
        filter: drop-shadow(0 0 30px rgba({glow},0.5));
        animation: float 3s ease-in-out infinite;
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}

    .brand-title {{
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFFFFF, var(--accent-light), var(--accent), #FFFFFF);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
        letter-spacing: -0.04em;
    }}

    @keyframes shimmer {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}

    .brand-sub {{
        font-size: 0.6rem;
        letter-spacing: 0.25em;
        color: var(--muted2) !important;
        font-weight: 800;
        margin-top: 3px;
    }}

    .brand-divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        margin: 15px 0;
        opacity: 0.3;
    }}

    /* ================================================
       ULTIMATE LUXURY CARDS
       ================================================ */
    .billionaire-card {{
        background: linear-gradient(145deg, rgba({glow},0.04), rgba(255,255,255,0.01)) !important;
        border: 1px solid rgba({glow},0.12) !important;
        border-radius: 24px !important;
        padding: 30px !important;
        box-shadow: 0 25px 70px rgba(0,0,0,0.4), 0 0 50px rgba({glow},0.03) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        position: relative;
        overflow: hidden;
    }}

    .billionaire-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba({glow},0.03), transparent 60%);
        pointer-events: none;
    }}

    .billionaire-card:hover {{
        border-color: rgba({glow},0.35) !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 35px 90px rgba(0,0,0,0.5), 0 0 60px rgba({glow},0.06) !important;
    }}

    /* ================================================
       METRIC CARDS - BILLIONAIRE
       ================================================ */
    .metric-billionaire {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, rgba({glow},0.05), rgba(255,255,255,0.01)) !important;
        border: 1px solid rgba({glow},0.15) !important;
        border-radius: 20px !important;
        padding: 28px 15px !important;
        text-align: center !important;
        min-height: 130px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.35) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }}

    .metric-billionaire::after {{
        content: '✦';
        position: absolute;
        top: 8px;
        right: 12px;
        font-size: 0.8rem;
        color: var(--accent);
        opacity: 0.2;
    }}

    .metric-billionaire:hover {{
        transform: translateY(-8px) scale(1.02) !important;
        border-color: rgba({glow},0.4) !important;
        box-shadow: 0 30px 70px rgba(0,0,0,0.5), 0 0 40px rgba({glow},0.08) !important;
    }}

    .metric-billionaire h3 {{
        color: var(--accent-light) !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 0 0 30px rgba({glow},0.15) !important;
    }}

    .metric-billionaire p {{
        color: #AEB7C5 !important;
        font-size: 0.7rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 8px !important;
    }}

    /* ================================================
       BILLIONAIRE BUTTONS
       ================================================ */
    .stButton > button, .stFormSubmitButton > button {{
        background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
        color: #07080C !important;
        -webkit-text-fill-color: #07080C !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        font-size: 1rem !important;
        min-height: 52px !important;
        padding: 0 2rem !important;
        box-shadow: 0 10px 35px rgba({glow},0.25) !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        letter-spacing: 0.03em !important;
        position: relative;
        overflow: hidden;
    }}

    .stButton > button::after, .stFormSubmitButton > button::after {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2), transparent 60%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}

    .stButton > button:hover::after, .stFormSubmitButton > button:hover::after {{
        opacity: 1;
    }}

    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 18px 50px rgba({glow},0.4) !important;
        filter: brightness(1.08);
    }}

    .stButton > button:active, .stFormSubmitButton > button:active {{
        transform: scale(0.95) !important;
    }}

    /* ================================================
       INPUTS - BILLIONAIRE
       ================================================ */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {{
        background: rgba(20, 25, 40, 0.9) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        caret-color: var(--accent-light) !important;
        border: 2px solid rgba({glow},0.15) !important;
        border-radius: 14px !important;
        min-height: 50px !important;
        font-weight: 600 !important;
        padding: 0 18px !important;
        transition: all 0.3s ease !important;
    }}

    .stTextInput input:hover, .stNumberInput input:hover {{
        border-color: rgba({glow},0.35) !important;
        background: rgba(25, 30, 50, 0.95) !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba({glow},0.15), 0 0 30px rgba({glow},0.05) !important;
        background: rgba(25, 30, 50, 0.95) !important;
    }}

    /* ================================================
       SIDEBAR CONTROLS
       ================================================ */
    .control-label {{
        color: var(--muted) !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.15em !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        margin-bottom: 5px !important;
    }}

    /* ================================================
       SCROLLBAR - LUXURY
       ================================================ */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}

    ::-webkit-scrollbar-track {{
        background: rgba(255,255,255,0.03);
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, var(--accent), var(--secondary));
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent-light);
    }}

    /* ================================================
       HERO - BILLIONAIRE
       ================================================ */
    .hero-billionaire {{
        text-align: center;
        padding: 30px 20px 50px 20px;
        animation: fadeInUp 0.8s ease;
    }}

    .hero-badge {{
        display: inline-block;
        padding: 10px 25px;
        border-radius: 999px;
        border: 1px solid rgba({glow},0.3);
        background: rgba({glow},0.05);
        color: var(--accent-light) !important;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        margin-bottom: 25px;
        box-shadow: 0 0 40px rgba({glow},0.05);
        backdrop-filter: blur(10px);
    }}

    .hero-billionaire h1 {{
        font-size: clamp(3.5rem, 10vw, 8rem) !important;
        font-weight: 900 !important;
        line-height: 1 !important;
        margin-bottom: 25px !important;
        background: linear-gradient(135deg, #FFFFFF 10%, var(--accent-bright) 40%, var(--accent) 70%, #FFFFFF 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 4s ease-in-out infinite;
        filter: drop-shadow(0 10px 40px rgba({glow},0.1));
    }}

    .hero-description {{
        max-width: 850px;
        margin: auto;
        font-size: 1.1rem;
        color: var(--muted) !important;
        line-height: 1.8;
    }}

    /* ================================================
       ANIMATIONS
       ================================================ */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba({glow},0.5); }}
        70% {{ box-shadow: 0 0 0 15px rgba({glow},0); }}
        100% {{ box-shadow: 0 0 0 0 rgba({glow},0); }}
    }}

    @keyframes glow {{
        0%, 100% {{ text-shadow: 0 0 20px rgba({glow},0.2); }}
        50% {{ text-shadow: 0 0 40px rgba({glow},0.4); }}
    }}

    .glow-text {{
        animation: glow 3s ease-in-out infinite;
    }}

    /* ================================================
       STATUS INDICATOR
       ================================================ */
    .status-billionaire {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 20px;
        border-radius: 16px;
        background: rgba({glow},0.04);
        border: 1px solid rgba({glow},0.12);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }}

    .status-dot {{
        width: 12px;
        height: 12px;
        background: var(--accent-light);
        border-radius: 50%;
        box-shadow: 0 0 20px rgba({glow},0.5);
        animation: pulse 2s infinite;
    }}

    .status-text {{
        color: var(--accent-light) !important;
        font-weight: 800;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }}

    /* ================================================
       PREDICTION RESULT - BILLIONAIRE
       ================================================ */
    .prediction-billionaire {{
        text-align: center;
        padding: 45px 30px;
        background: radial-gradient(circle at center, rgba({glow},0.08), transparent 70%);
        border: 2px solid rgba({glow},0.2);
        border-radius: 30px;
        box-shadow: 0 30px 100px rgba(0,0,0,0.4), 0 0 60px rgba({glow},0.04);
        position: relative;
        overflow: hidden;
    }}

    .prediction-billionaire::before {{
        content: '✦';
        position: absolute;
        top: 15px;
        right: 20px;
        font-size: 1.5rem;
        color: var(--accent);
        opacity: 0.15;
    }}

    .prediction-species {{
        font-size: clamp(2.5rem, 6vw, 4.5rem) !important;
        font-weight: 900 !important;
        color: var(--accent-light) !important;
        margin-bottom: 10px !important;
        text-shadow: 0 0 40px rgba({glow},0.15) !important;
    }}

    .prediction-confidence {{
        font-size: 1.3rem !important;
        color: var(--accent-bright) !important;
        font-weight: 800 !important;
    }}

    /* ================================================
       FEATURE GRID - BILLIONAIRE
       ================================================ */
    .feature-grid-billionaire {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 20px;
        margin: 30px 0;
    }}

    .feature-item-billionaire {{
        background: linear-gradient(145deg, rgba({glow},0.04), rgba(255,255,255,0.01));
        border: 1px solid rgba({glow},0.08);
        border-radius: 22px;
        padding: 30px 20px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        min-height: 220px;
        position: relative;
        overflow: hidden;
    }}

    .feature-item-billionaire::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}

    .feature-item-billionaire:hover::after {{
        opacity: 1;
    }}

    .feature-item-billionaire:hover {{
        transform: translateY(-8px) !important;
        border-color: rgba({glow},0.25) !important;
        box-shadow: 0 25px 60px rgba(0,0,0,0.4), 0 0 30px rgba({glow},0.04) !important;
    }}

    .feature-icon-billionaire {{
        font-size: 3rem;
        margin-bottom: 15px;
        filter: drop-shadow(0 0 20px rgba({glow},0.1));
    }}

    .feature-title-billionaire {{
        color: var(--accent-light) !important;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 10px;
    }}

    .feature-desc-billionaire {{
        color: #AAB2C0 !important;
        font-size: 0.9rem;
        line-height: 1.7;
    }}

    /* ================================================
       FOOTER - BILLIONAIRE
       ================================================ */
    .footer-billionaire {{
        text-align: center;
        margin-top: 80px;
        padding: 40px 20px;
        border-top: 1px solid rgba({glow},0.08);
        color: #596271 !important;
        font-size: 0.8rem;
    }}

    .footer-billionaire .brand {{
        color: var(--accent-light) !important;
        font-weight: 900;
        font-size: 1rem;
        letter-spacing: 0.1em;
    }}

    /* ================================================
       RESPONSIVE
       ================================================ */
    @media (max-width: 768px) {{
        .block-container {{ padding: 1rem !important; }}
        .metric-billionaire {{ padding: 18px 10px !important; min-height: 100px; }}
        .metric-billionaire h3 {{ font-size: 1.8rem !important; }}
        .feature-grid-billionaire {{ grid-template-columns: 1fr; }}
        .billionaire-card {{ padding: 20px !important; }}
        .hero-billionaire {{ padding: 15px 10px 30px !important; }}
        .
