"""
IrisAI – Ultra-Luxury / Trillionaire ML Classification Platform
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
# PASSWORD PROTECTION
# ============================================================

def check_password():

    def password_entered():

        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]

        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:

        st.markdown(
            """
            <div style="
                max-width:520px;
                margin:100px auto;
                padding:45px;
                background: rgba(17, 19, 27, 0.85);
                border:1px solid rgba(212,175,55,0.6);
                border-radius:24px;
                text-align:center;
                box-shadow:0 0 50px rgba(212,175,55,0.25), 0 20px 70px rgba(0,0,0,0.85);
                backdrop-filter: blur(20px);
            ">

                <div style="
                    font-size:65px;
                    margin-bottom:10px;
                    filter: drop-shadow(0 0 15px rgba(212,175,55,0.6));
                ">
                    🌸
                </div>

                <h1 style="
                    color:#F4D06F !important;
                    margin-bottom:5px;
                    text-shadow: 0 0 15px rgba(244, 208, 111, 0.5);
                ">
                    IrisAI
                </h1>

                <p style="
                    color:#B8C0CC !important;
                    font-size:15px;
                    letter-spacing: 1px;
                ">
                    Ultra-Luxury Machine Learning Platform
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_input(
            "🔐 Enter Access Password",
            type="password",
            on_change=password_entered,
            key="password"
        )

        return False

    elif not st.session_state["password_correct"]:

        st.markdown(
            """
            <div style="
                max-width:520px;
                margin:100px auto 30px auto;
                padding:40px;
                background: rgba(17, 19, 27, 0.85);
                border:1px solid rgba(255,92,92,0.6);
                border-radius:24px;
                text-align:center;
                box-shadow:0 0 50px rgba(255,92,92,0.25), 0 20px 70px rgba(0,0,0,0.85);
                backdrop-filter: blur(20px);
            ">

                <div style="font-size:55px;">
                    🔒
                </div>

                <h1 style="
                    color:#F4D06F !important;
                ">
                    IrisAI
                </h1>

                <p style="
                    color:#B8C0CC !important;
                ">
                    Authentication Required
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_input(
            "🔐 Enter Access Password",
            type="password",
            on_change=password_entered,
            key="password"
        )

        st.error("❌ Incorrect password")

        return False

    else:
        return True


if not check_password():
    st.stop()


# ============================================================
# FONT & THEME SELECTION
# ============================================================

if "font_choice" not in st.session_state:
    st.session_state.font_choice = "Inter"

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Trillionaire Gold"

if "animated_colors" not in st.session_state:
    st.session_state.animated_colors = True

font_options = {
    "Inter": "'Inter', sans-serif",
    "Poppins": "'Poppins', sans-serif",
    "Montserrat": "'Montserrat', sans-serif",
    "Roboto": "'Roboto', sans-serif",
    "Playfair Display": "'Playfair Display', serif",
    "Cormorant Garamond": "'Cormorant Garamond', serif",
    "Space Grotesk": "'Space Grotesk', sans-serif",
}

theme_palettes = {
    "Trillionaire Gold": {
        "primary": "#D4AF37",
        "primary_light": "#F4D06F",
        "primary_bright": "#FFE9A3",
        "accent": "#FFD700",
        "glow": "rgba(212,175,55,0.4)",
        "bg_radial_1": "rgba(212,175,55,0.12)",
        "bg_radial_2": "rgba(180,140,40,0.08)",
    },
    "Cyberpunk Neon": {
        "primary": "#00F0FF",
        "primary_light": "#70F6FF",
        "primary_bright": "#B3FCFF",
        "accent": "#FF007F",
        "glow": "rgba(0,240,255,0.4)",
        "bg_radial_1": "rgba(0,240,255,0.12)",
        "bg_radial_2": "rgba(255,0,127,0.08)",
    },
    "Imperial Emerald": {
        "primary": "#00E676",
        "primary_light": "#69F0AE",
        "primary_bright": "#B9F6CA",
        "accent": "#00B0FF",
        "glow": "rgba(0,230,118,0.4)",
        "bg_radial_1": "rgba(0,230,118,0.12)",
        "bg_radial_2": "rgba(0,176,255,0.08)",
    },
    "Electric Sapphire": {
        "primary": "#2979FF",
        "primary_light": "#82B1FF",
        "primary_bright": "#E3F2FD",
        "accent": "#7C4DFF",
        "glow": "rgba(41,121,255,0.4)",
        "bg_radial_1": "rgba(41,121,255,0.12)",
        "bg_radial_2": "rgba(124,77,255,0.08)",
    },
    "Royal Amethyst": {
        "primary": "#D500F9",
        "primary_light": "#E040FB",
        "primary_bright": "#EA80FC",
        "accent": "#651FFF",
        "glow": "rgba(213,0,249,0.4)",
        "bg_radial_1": "rgba(213,0,249,0.12)",
        "bg_radial_2": "rgba(101,31,255,0.08)",
    },
    "Rose Gold Luxury": {
        "primary": "#B76E79",
        "primary_light": "#E8C5C8",
        "primary_bright": "#F4E3E5",
        "accent": "#FFB6C1",
        "glow": "rgba(183,110,121,0.4)",
        "bg_radial_1": "rgba(183,110,121,0.12)",
        "bg_radial_2": "rgba(255,182,193,0.08)",
    },
}

selected_font = st.session_state.font_choice
selected_theme = st.session_state.theme_choice
current_palette = theme_palettes[selected_theme]


# ============================================================
# ULTRA LUXURY DYNAMIC CSS ENGINE
# ============================================================

def load_css(font_name, palette, is_animated):

    font_family = font_options[font_name]
    p = palette

    animation_css = ""
    if is_animated:
        animation_css = """
        @keyframes hueShift {
            0% { filter: hue-rotate(0deg); }
            50% { filter: hue-rotate(45deg); }
            100% { filter: hue-rotate(0deg); }
        }
        .live-chroma {
            animation: hueShift 10s infinite alternate ease-in-out;
        }
        """

    st.markdown(
        f"""
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&family=Montserrat:wght@400;500;600;700;800&family=Roboto:wght@400;500;700&family=Playfair+Display:wght@400;500;600;700&family=Cormorant+Garamond:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        {animation_css}

        :root {{
            --bg: #030407;
            --bg2: #07090E;
            --panel: #0D1017;
            --panel2: #121621;
            --border: {p["primary"]}44;
            --gold: {p["primary"]};
            --gold-light: {p["primary_light"]};
            --gold-bright: {p["primary_bright"]};
            --accent: {p["accent"]};
            --glow: {p["glow"]};
            --text: #F8FAFC;
            --muted: #AAB2C0;
            --muted2: #7F8998;
            --green: #00E676;
            --red: #FF5252;
            --blue: #448AFF;
        }}

        html, body, [class*="css"], .stApp {{
            font-family: {font_family} !important;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 10% 0%, {p["bg_radial_1"]}, transparent 40%),
                radial-gradient(circle at 90% 90%, {p["bg_radial_2"]}, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(15,18,28,0.5), transparent 80%),
                var(--bg) !important;
            background-attachment: fixed !important;
            color: var(--text) !important;
        }}

        /* Dynamic background particle mesh simulator */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: radial-gradient(2px 2px at 20px 30px, var(--gold-light), rgba(0,0,0,0)),
                        radial-gradient(2px 2px at 40px 70px, var(--accent), rgba(0,0,0,0)),
                        radial-gradient(1px 1px at 90px 40px, #ffffff, rgba(0,0,0,0));
            background-repeat: repeat;
            background-size: 150px 150px;
            opacity: 0.12;
            pointer-events: none;
            z-index: 0;
        }}

        .main {{
            background: transparent !important;
            position: relative;
            z-index: 1;
        }}

        .block-container {{
            max-width: 1500px !important;
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
        }}

        /* ====================================================
           GLOBAL TEXT VISIBILITY & GLOWS
           ==================================================== */

        h1, h2, h3, h4, h5, h6,
        p, span, label, li, strong, small,
        .stMarkdown, .stText,
        [data-testid="stMarkdownContainer"] {{
            color: var(--text) !important;
        }}

        h1 {{
            font-size: clamp(2rem, 4vw, 3.4rem) !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }}

        h2 {{
            font-weight: 750 !important;
            color: var(--gold-light) !important;
        }}

        p {{
            color: var(--muted) !important;
        }}

        /* ====================================================
           SIDEBAR ULTRA LUXURY
           ==================================================== */

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #090B10 0%, #040508 100%) !important;
            border-right: 1px solid var(--border) !important;
            box-shadow: 10px 0 30px rgba(0,0,0,0.5) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text) !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: var(--border) !important;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            color: #E9EDF3 !important;
            font-weight: 600 !important;
            transition: all 0.2s ease;
            padding: 4px 8px;
            border-radius: 8px;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            color: var(--gold-light) !important;
            background: rgba(255,255,255,0.03);
            box-shadow: 0 0 12px var(--glow);
        }}

        /* ====================================================
           PREMIUM LUXURY CARDS
           ==================================================== */

        .luxury-card {{
            background: linear-gradient(145deg, rgba(20, 24, 35, 0.7), rgba(10, 12, 18, 0.8)) !important;
            border: 1px solid var(--border) !important;
            border-radius: 22px !important;
            padding: 26px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08) !important;
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            margin-bottom: 20px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}

        .luxury-card:hover {{
            border-color: var(--gold) !important;
            transform: translateY(-3px);
            box-shadow: 0 25px 75px rgba(0,0,0,0.7), 0 0 35px var(--glow) !important;
        }}

        /* ====================================================
           METRIC CARDS
           ==================================================== */

        .metric-card {{
            background: linear-gradient(145deg, #121622, #0A0C12) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            padding: 24px 15px !important;
            text-align: center !important;
            min-height: 120px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.4);
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .metric-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--gold), transparent);
        }}

        .metric-card:hover {{
            border-color: var(--gold-light) !important;
            transform: translateY(-4px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.6), 0 0 25px var(--glow);
        }}

        .metric-card h3 {{
            color: var(--gold-light) !important;
            font-size: 2.3rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            text-shadow: 0 0 15px var(--glow);
        }}

        .metric-card p {{
            color: #AEB7C5 !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}

        /* ====================================================
           INPUTS & SELECTORS
           ==================================================== */

        .stTextInput input, .stNumberInput input {{
            background: #0E121A !important;
            color: #FFFFFF !important;
            caret-color: var(--gold-light) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            transition: all 0.25s ease;
        }}

        .stTextInput input:focus, .stNumberInput input:focus {{
            background: #131824 !important;
            color: #FFFFFF !important;
            border: 1px solid var(--gold) !important;
            box-shadow: 0 0 20px var(--glow) !important;
        }}

        .stTextInput input::placeholder {{
            color: #727C8B !important;
        }}

        .stNumberInput button {{
            background: #181D2A !important;
            color: var(--gold-light) !important;
            border: none !important;
        }}

        .stNumberInput button:hover {{
            background: #232A3C !important;
            box-shadow: 0 0 10px var(--glow);
        }}

        .stSelectbox div[data-baseweb="select"] > div {{
            background: #0E121A !important;
            color: #FFFFFF !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }}

        .stSelectbox div[data-baseweb="select"] span {{
            color: #FFFFFF !important;
        }}

        div[data-baseweb="popover"], div[data-baseweb="menu"] {{
            background: #0D1017 !important;
            border: 1px solid var(--border) !important;
        }}

        div[data-baseweb="menu"] * {{
            color: #FFFFFF !important;
        }}

        /* ====================================================
           BUTTONS & TOGGLES
           ==================================================== */

        .stButton > button, .stFormSubmitButton > button {{
            background: linear-gradient(135deg, var(--gold), var(--gold-light)) !important;
            color: #040507 !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            min-height: 48px !important;
            letter-spacing: 0.03em;
            box-shadow: 0 8px 25px var(--glow) !important;
            transition: all 0.3s ease !important;
        }}

        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            transform: translateY(-2px) scale(1.01) !important;
            box-shadow: 0 12px 35px var(--glow), 0 0 20px var(--gold-bright) !important;
        }}

        .stDownloadButton > button {{
            background: linear-gradient(135deg, var(--gold), var(--gold-light)) !important;
            color: #040507 !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
        }}

        /* ====================================================
           DATAFRAME & EXPANDERS
           ==================================================== */

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            background: #0B0E14 !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}

        [data-testid="stDataFrame"] * {{
            color: #F5F7FA !important;
        }}

        [data-testid="stExpander"] {{
            background: #0D1017 !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}

        [data-testid="stExpander"] * {{
            color: #F5F7FA !important;
        }}

        [data-testid="stAlert"] {{
            background: #121622 !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
        }}

        [data-testid="stAlert"] * {{
            color: #FFFFFF !important;
        }}

        /* ====================================================
           FEATURE CARDS
           ==================================================== */

        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}

        .feature-item {{
            background: linear-gradient(145deg, #121622, #0A0C12);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            min-height: 190px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}

        .feature-item:hover {{
            transform: translateY(-6px);
            border-color: var(--gold);
            box-shadow: 0 18px 50px rgba(0,0,0,0.6), 0 0 30px var(--glow);
        }}

        .feature-icon {{
            font-size: 2.7rem;
            margin-bottom: 12px;
            filter: drop-shadow(0 0 10px var(--glow));
        }}

        .feature-title {{
            color: var(--gold-light) !important;
            font-size: 1.1rem;
            font-weight: 750;
            margin-bottom: 8px;
        }}

        .feature-desc {{
            color: #AAB2C0 !important;
            font-size: 0.88rem;
            line-height: 1.65;
        }}

        /* ====================================================
           HERO
           ==================================================== */

        .hero {{
            text-align: center;
            padding: 40px 20px 50px;
            animation: fadeIn 0.7s ease;
        }}

        .hero h1 {{
            background: linear-gradient(135deg, #FFFFFF, var(--gold-light), var(--gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: clamp(2.5rem, 6vw, 4.8rem) !important;
            text-shadow: 0 0 30px var(--glow);
        }}

        .hero p {{
            max-width: 800px;
            margin: auto;
            font-size: 1.1rem;
            color: #AAB2C0 !important;
            letter-spacing: 0.02em;
        }}

        /* ====================================================
           FOOTER
           ==================================================== */

        .footer {{
            text-align: center;
            margin-top: 60px;
            padding: 25px;
            border-top: 1px solid var(--border);
            color: #6F7887 !important;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(15px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @media (max-width: 768px) {{
            .block-container {{ padding: 1rem !important; }}
            .hero {{ padding: 20px 10px 30px; }}
            .metric-card h3 {{ font-size: 1.6rem !important; }}
            .feature-grid {{ grid-template-columns: 1fr; }}
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


load_css(selected_font, current_palette, st.session_state.animated_colors)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_iris_data():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = iris.target

    df["species_name"] = df["species"].apply(
        lambda x: iris.target_names[x]
    )

    return (
        df,
        iris.target_names,
        iris.feature_names
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
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    random_state=42
                )
            )
        ]),

        "Decision Tree": Pipeline([
            (
                "clf",
                DecisionTreeClassifier(
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]),

        "K-Nearest Neighbors": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ]),

        "Support Vector Machine": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                SVC(
                    probability=True,
                    random_state=42
                )
            )
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
                y_pred
            ),

            "Precision": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "Confusion Matrix": confusion_matrix(
                y_test,
                y_pred
            )
        }

        trained_pipelines[name] = pipe

    best_model_name = max(
        results,
        key=lambda k: results[k]["Accuracy"]
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
        y_test
    )


(
    pipelines,
    model_results,
    best_model_name,
    best_pipeline,
    X_test,
    y_test
) = train_models()


# ============================================================
# SESSION STATE
# ============================================================

if "best_model" not in st.session_state:
    st.session_state.best_model = best_pipeline

if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = best_model_name

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "history" not in st.session_state:
    st.session_state.history = []

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"


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
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    f"""
    <div style="
        text-align:center;
        padding:10px 0 20px;
    ">

        <div class="live-chroma" style="
            font-size:48px;
            filter: drop-shadow(0 0 15px var(--glow));
        ">
            🌸
        </div>

        <h1 class="live-chroma" style="
            color:var(--gold-light) !important;
            font-size:2.1rem !important;
            margin:0;
            text-shadow: 0 0 15px var(--glow);
        ">
            IrisAI
        </h1>

        <p style="
            color:var(--muted2) !important;
            margin-top:4px;
            letter-spacing:0.15em;
            font-size:0.75rem;
            font-weight:700;
        ">
            TRILLIONAIRE ML PLATFORM
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


# ============================================================
# DYNAMIC COLOR & FONT CONTROLS
# ============================================================

st.sidebar.markdown(
    """
    <p style="
        color:var(--gold-light) !important;
        font-weight:700;
        letter-spacing:0.04em;
    ">
        👑 LUXURY PALETTE
    </p>
    """,
    unsafe_allow_html=True
)

theme_selection = st.sidebar.selectbox(
    "Choose Theme Palette",
    list(theme_palettes.keys()),
    index=list(theme_palettes.keys()).index(
        st.session_state.theme_choice
    )
)

if theme_selection != st.session_state.theme_choice:
    st.session_state.theme_choice = theme_selection
    st.rerun()

st.sidebar.markdown(
    """
    <p style="
        color:var(--gold-light) !important;
        font-weight:700;
        letter-spacing:0.04em;
        margin-top:15px;
    ">
        ✦ INTERFACE FONT
    </p>
    """,
    unsafe_allow_html=True
)

font_selection = st.sidebar.selectbox(
    "Choose your font",
    list(font_options.keys()),
    index=list(font_options.keys()).index(
        st.session_state.font_choice
    )
)

if font_selection != st.session_state.font_choice:
    st.session_state.font_choice = font_selection
    st.rerun()

st.session_state.animated_colors = st.sidebar.toggle(
    "✨ Live Color Shifts",
    value=st.session_state.animated_colors
)


st.sidebar.markdown(
    f"""
    <div style="
        background:#0D1017;
        border:1px solid var(--border);
        border-radius:12px;
        padding:12px;
        margin:10px 0 20px;
        text-align:center;
        box-shadow: 0 0 15px var(--glow);
    ">

        <span style="
            color:var(--muted2) !important;
            font-size:11px;
            letter-spacing:1px;
        ">
            ACTIVE CONFIG
        </span>

        <br>

        <strong style="
            color:var(--gold-light) !important;
            font-size:13px;
        ">
            {theme_selection} | {font_selection}
        </strong>

    </div>
    """,
    unsafe_allow_html=True
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
    "📚 About Project",
]

selected_page = st.sidebar.radio(
    "📌 NAVIGATION",
    nav_options,
    index=(
        nav_options.index(st.session_state.page)
        if st.session_state.page in nav_options
        else 0
    )
)

if selected_page != st.session_state.page:

    st.session_state.page = selected_page

    st.rerun()


st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <div style="
        text-align:center;
        padding:10px;
    ">

        <span style="
            color:var(--muted2) !important;
            font-size:11px;
            text-transform:uppercase;
            letter-spacing:1px;
        ">
            Best Model
        </span>

        <div style="
            color:var(--gold-light) !important;
            font-size:14px;
            font-weight:700;
            margin-top:5px;
            text-shadow: 0 0 10px var(--glow);
        ">
            {best_model_name}
        </div>

    </div>
    """,
    unsafe_allow_html=True
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
        unsafe_allow_html=True
    )


def navigate_to(page_name):

    st.session_state.page = page_name
    st.rerun()


def luxury_back_button():

    if st.button(
        "← Back to Home",
        key="back_home",
        use_container_width=False
    ):
        navigate_to("🏠 Home")


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "🏠 Home":

    current_accuracy = model_results[
        best_model_name
    ]["Accuracy"]

    st.markdown(
        """
        <div class="hero">

            <h1 class="live-chroma">
                🌸 IrisAI
            </h1>

            <p>
                The Future of Flower Classification —
                AI-Powered, Real-Time & Ultra-Intelligent.
            </p>

        </div>
        """,
        unsafe_allow_html=True
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
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{current_accuracy:.1%}</h3>
                <p>Best Accuracy</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
                <h3>5</h3>
                <p>ML Algorithms</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{st.session_state.prediction_count}</h3>
                <p>Predictions</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "## 🚀 Why IrisAI?"
    )

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
                    The highest-performing model
                    is selected automatically.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">🔬</div>
                <div class="feature-title">
                    Explainable AI
                </div>
                <div class="feature-desc">
                    Explore feature importance and
                    model behaviour.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-title">
                    Interactive Analytics
                </div>
                <div class="feature-desc">
                    Explore the dataset through
                    interactive visualizations.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">
                    Instant Predictions
                </div>
                <div class="feature-desc">
                    Enter flower measurements and
                    receive an instant prediction.
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">💎</div>
                <div class="feature-title">
                    Trillionaire Experience
                </div>
                <div class="feature-desc">
                    Designed with a high-end
                    technology aesthetic and dynamic live colors.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🎯 Quick Access")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🤖 AI Prediction",
            use_container_width=True
        ):
            navigate_to("🤖 AI Prediction")

    with col2:

        if st.button(
            "📊 Dataset Explorer",
            use_container_width=True
        ):
            navigate_to("📊 Dataset Explorer")

    with col3:

        if st.button(
            "📈 Data Visualization",
            use_container_width=True
        ):
            navigate_to("📈 Data Visualization")


# Footer tag across platform
st.markdown(
    f"""
    <div class="footer">
        IrisAI Ultra-Luxury ML Platform &bull; Active Theme: <strong>{selected_theme}</strong> &bull; Powered by Streamlit & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)
