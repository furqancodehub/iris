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
                background:#11131B;
                border:1px solid rgba(212,175,55,0.35);
                border-radius:24px;
                text-align:center;
                box-shadow:0 20px 70px rgba(0,0,0,0.65);
            ">

                <div style="
                    font-size:60px;
                    margin-bottom:10px;
                ">
                    🌸
                </div>

                <h1 style="
                    color:#F4D06F !important;
                    margin-bottom:5px;
                ">
                    IrisAI
                </h1>

                <p style="
                    color:#B8C0CC !important;
                    font-size:15px;
                ">
                    Premium Machine Learning Platform
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
                background:#11131B;
                border:1px solid rgba(212,175,55,0.35);
                border-radius:24px;
                text-align:center;
                box-shadow:0 20px 70px rgba(0,0,0,0.65);
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
# FONT SELECTION
# ============================================================

if "font_choice" not in st.session_state:
    st.session_state.font_choice = "Inter"

font_options = {
    "Inter": "'Inter', sans-serif",
    "Poppins": "'Poppins', sans-serif",
    "Montserrat": "'Montserrat', sans-serif",
    "Roboto": "'Roboto', sans-serif",
    "Playfair Display": "'Playfair Display', serif",
    "Cormorant Garamond": "'Cormorant Garamond', serif",
    "Space Grotesk": "'Space Grotesk', sans-serif",
}

selected_font = st.session_state.font_choice


# ============================================================
# PREMIUM CSS
# ============================================================

def load_css(font_name):

    font_family = font_options[font_name]

    st.markdown(
        f"""
        <style>

        @import url('https://fonts.googleapis.com/css2?
        family=Inter:wght@300;400;500;600;700;800&
        family=Poppins:wght@400;500;600;700;800&
        family=Montserrat:wght@400;500;600;700;800&
        family=Roboto:wght@400;500;700&
        family=Playfair+Display:wght@400;500;600;700&
        family=Cormorant+Garamond:wght@400;500;600;700&
        family=Space+Grotesk:wght@400;500;600;700&
        display=swap');

        :root {{
            --bg: #07080C;
            --bg2: #0B0D12;
            --panel: #11141B;
            --panel2: #151923;
            --border: rgba(212,175,55,0.25);
            --gold: #D4AF37;
            --gold-light: #F4D06F;
            --gold-bright: #FFE9A3;
            --text: #F8FAFC;
            --muted: #AAB2C0;
            --muted2: #7F8998;
            --green: #43D17A;
            --red: #FF5C5C;
            --blue: #5EA7FF;
        }}

        html,
        body,
        [class*="css"],
        .stApp {{
            font-family: {font_family} !important;
        }}

        .stApp {{
            background:
                radial-gradient(
                    circle at 10% 0%,
                    rgba(212,175,55,0.08),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 10%,
                    rgba(80,110,180,0.07),
                    transparent 30%
                ),
                var(--bg) !important;

            color: var(--text) !important;
        }}

        .main {{
            background: transparent !important;
        }}

        .block-container {{
            max-width: 1500px !important;
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
        }}

        /* ====================================================
           GLOBAL TEXT VISIBILITY
           ==================================================== */

        h1, h2, h3, h4, h5, h6,
        p, span, label, li, strong, small,
        .stMarkdown,
        .stText,
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
        }}

        p {{
            color: var(--muted) !important;
        }}

        /* ====================================================
           SIDEBAR
           ==================================================== */

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    #0D0F15 0%,
                    #090A0E 100%
                ) !important;

            border-right:
                1px solid rgba(212,175,55,0.18) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text) !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: rgba(212,175,55,0.18) !important;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            color: #E9EDF3 !important;
            font-weight: 600 !important;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            color: var(--gold-light) !important;
        }}

        /* ====================================================
           PREMIUM CARDS
           ==================================================== */

        .luxury-card {{
            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.055),
                    rgba(255,255,255,0.018)
                ) !important;

            border:
                1px solid rgba(212,175,55,0.20) !important;

            border-radius: 22px !important;

            padding: 26px !important;

            box-shadow:
                0 20px 60px rgba(0,0,0,0.38),
                inset 0 1px 0 rgba(255,255,255,0.035) !important;

            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);

            margin-bottom: 20px;
        }}

        .luxury-card:hover {{
            border-color:
                rgba(212,175,55,0.42) !important;

            box-shadow:
                0 25px 75px rgba(0,0,0,0.50),
                0 0 30px rgba(212,175,55,0.05) !important;
        }}

        /* ====================================================
           METRIC CARDS
           ==================================================== */

        .metric-card {{
            background:
                linear-gradient(
                    145deg,
                    #141720,
                    #0E1016
                ) !important;

            border:
                1px solid rgba(212,175,55,0.22) !important;

            border-radius: 18px !important;

            padding: 24px 15px !important;

            text-align: center !important;

            min-height: 120px;

            box-shadow:
                0 15px 45px rgba(0,0,0,0.35);
        }}

        .metric-card h3 {{
            color: var(--gold-light) !important;

            font-size: 2.1rem !important;

            font-weight: 800 !important;

            margin: 0 !important;
        }}

        .metric-card p {{
            color: #AEB7C5 !important;

            font-size: 0.78rem !important;

            font-weight: 600 !important;

            letter-spacing: 0.08em;

            text-transform: uppercase;
        }}

        /* ====================================================
           INPUTS
           ==================================================== */

        .stTextInput input,
        .stNumberInput input {{
            background: #151922 !important;

            color: #FFFFFF !important;

            caret-color: #F4D06F !important;

            border:
                1px solid rgba(212,175,55,0.22) !important;

            border-radius: 12px !important;
        }}

        .stTextInput input:focus,
        .stNumberInput input:focus {{
            background: #191D27 !important;

            color: #FFFFFF !important;

            border:
                1px solid #D4AF37 !important;

            box-shadow:
                0 0 0 3px rgba(212,175,55,0.12) !important;
        }}

        .stTextInput input::placeholder {{
            color: #727C8B !important;
        }}

        .stNumberInput button {{
            background: #1C212C !important;

            color: #F4D06F !important;

            border: none !important;
        }}

        .stNumberInput button:hover {{
            background: #282E3A !important;
        }}

        /* ====================================================
           SELECT BOX
           ==================================================== */

        .stSelectbox div[data-baseweb="select"] > div {{
            background: #151922 !important;

            color: #FFFFFF !important;

            border:
                1px solid rgba(212,175,55,0.22) !important;

            border-radius: 12px !important;
        }}

        .stSelectbox div[data-baseweb="select"] span {{
            color: #FFFFFF !important;
        }}

        /* Dropdown popup */
        div[data-baseweb="popover"] {{
            background: #11141B !important;
        }}

        div[data-baseweb="menu"] {{
            background: #11141B !important;
        }}

        div[data-baseweb="menu"] * {{
            color: #FFFFFF !important;
        }}

        /* ====================================================
           BUTTONS
           ==================================================== */

        .stButton > button,
        .stFormSubmitButton > button {{
            background:
                linear-gradient(
                    135deg,
                    #D4AF37,
                    #F4D06F
                ) !important;

            color: #08090C !important;

            border: none !important;

            border-radius: 12px !important;

            font-weight: 800 !important;

            min-height: 45px !important;

            box-shadow:
                0 8px 25px rgba(212,175,55,0.15) !important;

            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            transform: translateY(-2px) !important;

            box-shadow:
                0 12px 35px rgba(212,175,55,0.28) !important;
        }}

        .stDownloadButton > button {{
            background:
                linear-gradient(
                    135deg,
                    #D4AF37,
                    #F4D06F
                ) !important;

            color: #08090C !important;

            border: none !important;

            border-radius: 12px !important;

            font-weight: 800 !important;
        }}

        /* ====================================================
           DATAFRAME
           ==================================================== */

        [data-testid="stDataFrame"] {{
            border:
                1px solid rgba(212,175,55,0.18) !important;

            border-radius: 14px !important;

            overflow: hidden !important;

            background: #10131A !important;
        }}

        [data-testid="stDataFrame"] * {{
            color: #F5F7FA !important;
        }}

        /* ====================================================
           TABS / EXPANDERS
           ==================================================== */

        [data-testid="stExpander"] {{
            background: #11141B !important;

            border:
                1px solid rgba(212,175,55,0.18) !important;

            border-radius: 14px !important;
        }}

        [data-testid="stExpander"] * {{
            color: #F5F7FA !important;
        }}

        /* ====================================================
           ALERTS
           ==================================================== */

        [data-testid="stAlert"] {{
            background: #151922 !important;

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

            grid-template-columns:
                repeat(auto-fit, minmax(250px, 1fr));

            gap: 18px;

            margin: 25px 0;
        }}

        .feature-item {{
            background:
                linear-gradient(
                    145deg,
                    #151821,
                    #0E1016
                );

            border:
                1px solid rgba(255,255,255,0.07);

            border-radius: 18px;

            padding: 25px;

            text-align: center;

            transition: all 0.25s ease;

            min-height: 190px;
        }}

        .feature-item:hover {{
            transform: translateY(-5px);

            border-color:
                rgba(212,175,55,0.35);

            box-shadow:
                0 18px 50px rgba(0,0,0,0.4);
        }}

        .feature-icon {{
            font-size: 2.5rem;

            margin-bottom: 12px;
        }}

        .feature-title {{
            color: #F4D06F !important;

            font-size: 1.05rem;

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

            padding: 35px 20px 45px;

            animation: fadeIn 0.7s ease;
        }}

        .hero h1 {{
            background:
                linear-gradient(
                    135deg,
                    #FFFFFF,
                    #F4D06F,
                    #D4AF37
                );

            -webkit-background-clip: text;

            -webkit-text-fill-color: transparent;

            font-size:
                clamp(2.4rem, 6vw, 4.5rem) !important;
        }}

        .hero p {{
            max-width: 800px;

            margin: auto;

            font-size: 1.05rem;

            color: #AAB2C0 !important;
        }}

        /* ====================================================
           FOOTER
           ==================================================== */

        .footer {{
            text-align: center;

            margin-top: 60px;

            padding: 25px;

            border-top:
                1px solid rgba(212,175,55,0.15);

            color: #6F7887 !important;

            font-size: 0.8rem;
        }}

        /* ====================================================
           ANIMATION
           ==================================================== */

        @keyframes fadeIn {{

            from {{
                opacity: 0;
                transform: translateY(15px);
            }}

            to {{
                opacity: 1;
                transform: translateY(0);
            }}

        }}

        /* ====================================================
           MOBILE
           ==================================================== */

        @media (max-width: 768px) {{

            .block-container {{
                padding: 1rem !important;
            }}

            .hero {{
                padding: 20px 10px 30px;
            }}

            .metric-card h3 {{
                font-size: 1.5rem !important;
            }}

            .feature-grid {{
                grid-template-columns: 1fr;
            }}

        }}

        </style>
        """,
        unsafe_allow_html=True
    )


load_css(selected_font)


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
    """
    <div style="
        text-align:center;
        padding:10px 0 20px;
    ">

        <div style="
            font-size:42px;
        ">
            🌸
        </div>

        <h1 style="
            color:#F4D06F !important;
            font-size:1.9rem !important;
            margin:0;
        ">
            IrisAI
        </h1>

        <p style="
            color:#7F8998 !important;
            margin-top:4px;
        ">
            PREMIUM ML PLATFORM
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


# ============================================================
# FONT CONTROL
# ============================================================

st.sidebar.markdown(
    """
    <p style="
        color:#F4D06F !important;
        font-weight:700;
        letter-spacing:0.04em;
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


st.sidebar.markdown(
    f"""
    <div style="
        background:#11141B;
        border:1px solid rgba(212,175,55,0.18);
        border-radius:12px;
        padding:12px;
        margin:10px 0 20px;
        text-align:center;
    ">

        <span style="
            color:#7F8998 !important;
            font-size:11px;
        ">
            ACTIVE FONT
        </span>

        <br>

        <strong style="
            color:#F4D06F !important;
        ">
            {font_selection}
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
            color:#7F8998 !important;
            font-size:11px;
            text-transform:uppercase;
        ">
            Best Model
        </span>

        <div style="
            color:#F4D06F !important;
            font-size:14px;
            font-weight:700;
            margin-top:5px;
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

            <h1>
                🌸 IrisAI
            </h1>

            <p>
                The Future of Flower Classification —
                AI-Powered, Real-Time & Intelligent.
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
                    Premium Experience
                </div>
                <div class="feature-desc">
                    Designed with a high-end
                    technology aesthetic.
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

        if st.button(
            "📊 Dataset Explorer",
            use_container_width=True
        ):
            navigate_to("📊 Dataset Explorer")

    with col2:

        if st.button(
            "📈 Data Visualization",
            use_container_width=True
        ):
            navigate_to("📈 Data Visualization")

        if st.button(
            "🧠 Model Performance",
            use_container_width=True
        ):
            navigate_to("🧠 Model Performance")

    with col3:

        if st.button(
            "🔬 Explainable AI",
            use_container_width=True
        ):
            navigate_to("🔬 Explainable AI")

        if st.button(
            "📚 About Project",
            use_container_width=True
        ):
            navigate_to("📚 About Project")


# ============================================================
# AI PREDICTION
# ============================================================

elif st.session_state.page == "🤖 AI Prediction":

    luxury_back_button()

    st.markdown(
        """
        <h1>🤖 Predict Iris Species</h1>

        <p>
            Enter the four measurements and let IrisAI
            classify the flower.
        </p>
        """,
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            sepal_len = st.number_input(
                "📏 Sepal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=5.1,
                step=0.1
            )

        with col2:
            sepal_wid = st.number_input(
                "📐 Sepal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=3.5,
                step=0.1
            )

        with col3:
            petal_len = st.number_input(
                "📏 Petal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=1.4,
                step=0.1
            )

        with col4:
            petal_wid = st.number_input(
                "📐 Petal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=0.2,
                step=0.1
            )

        submitted = st.form_submit_button(
            "🔮 Predict Species",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                sepal_len,
                sepal_wid,
                petal_len,
                petal_wid
            ]],
            columns=feature_names
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

        confidence = np.max(
            probabilities
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        record = {
            "species": pred_species,
            "confidence": confidence,
            "inputs": [
                sepal_len,
                sepal_wid,
                petal_len,
                petal_wid
            ],
            "timestamp": timestamp
        }

        st.session_state.prediction_count += 1

        st.session_state.last_prediction = record

        st.session_state.history.append(record)

        luxury_card(
            f"""
            <div style="text-align:center;">

                <div style="
                    font-size:50px;
                ">
                    🌸
                </div>

                <h2 style="
                    color:#F4D06F !important;
                ">
                    {pred_species}
                </h2>

                <p style="
                    color:#FFFFFF !important;
                    font-size:1.1rem;
                ">
                    Confidence:
                    <strong style="
                        color:#F4D06F !important;
                    ">
                        {confidence:.1%}
                    </strong>
                </p>

            </div>
            """
        )

        prob_df = pd.DataFrame({
            "Species": target_names,
            "Probability": probabilities
        }).sort_values(
            "Probability"
        )

        fig = px.bar(
            prob_df,
            x="Probability",
            y="Species",
            orientation="h",
            color="Species",
            title="Prediction Probabilities"
        )

        fig.update_layout(
            paper_bgcolor="#0B0D12",
            plot_bgcolor="#0B0D12",
            font=dict(
                color="#FFFFFF"
            ),
            title_font=dict(
                color="#F4D06F"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if st.session_state.last_prediction:

        last = st.session_state.last_prediction

        st.markdown(
            "### 📋 Last Prediction"
        )

        st.info(
            f"🌸 {last['species']}  •  "
            f"Confidence: {last['confidence']:.2%}  •  "
            f"{last['timestamp']}"
        )

    if st.session_state.history:

        with st.expander(
            "📜 Prediction History"
        ):

            history_df = pd.DataFrame(
                st.session_state.history
            )

            st.dataframe(
                history_df,
                use_container_width=True
            )

            csv = history_df.to_csv(
                index=False
            ).encode()

            st.download_button(
                "📥 Download History",
                csv,
                "prediction_history.csv",
                "text/csv",
                use_container_width=True
            )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif st.session_state.page == "📊 Dataset Explorer":

    luxury_back_button()

    st.markdown(
        "<h1>📊 Dataset Explorer</h1>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    search = st.text_input(
        "🔍 Search species"
    )

    if search:

        filtered = df[
            df["species_name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    else:

        filtered = df

    st.dataframe(
        filtered,
        use_container_width=True
    )

    st.markdown(
        "## 📈 Descriptive Statistics"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown(
        "## 🌸 Class Distribution"
    )

    counts = (
        df["species_name"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Species",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Species",
        y="Count",
        color="Species",
        title="Samples per Species"
    )

    fig.update_layout(
        paper_bgcolor="#0B0D12",
        plot_bgcolor="#0B0D12",
        font=dict(color="#FFFFFF")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DATA VISUALIZATION
# ============================================================

elif st.session_state.page == "📈 Data Visualization":

    luxury_back_button()

    st.markdown(
        "<h1>📈 Data Visualization</h1>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        x_feature = st.selectbox(
            "X-axis",
            feature_names
        )

    with col2:

        y_feature = st.selectbox(
            "Y-axis",
            feature_names,
            index=2
        )

    scatter = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="species_name",
        title=f"{x_feature} vs {y_feature}",
        hover_data=feature_names
    )

    scatter.update_layout(
        paper_bgcolor="#0B0D12",
        plot_bgcolor="#0B0D12",
        font=dict(color="#FFFFFF")
    )

    st.plotly_chart(
        scatter,
        use_container_width=True
    )

    st.markdown(
        "## 📊 Feature Distribution"
    )

    selected_feature = st.selectbox(
        "Select Feature",
        feature_names,
        key="distribution"
    )

    hist = px.histogram(
        df,
        x=selected_feature,
        color="species_name",
        marginal="box",
        title=f"Distribution of {selected_feature}"
    )

    hist.update_layout(
        paper_bgcolor="#0B0D12",
        plot_bgcolor="#0B0D12",
        font=dict(color="#FFFFFF")
    )

    st.plotly_chart(
        hist,
        use_container_width=True
    )

    st.markdown(
        "## 🔥 Feature Correlation"
    )

    correlation = df[
        feature_names
    ].corr()

    corr_fig = px.imshow(
        correlation,
        text_auto=True,
        title="Feature Correlation Matrix"
    )

    corr_fig.update_layout(
        paper_bgcolor="#0B0D12",
        font=dict(color="#FFFFFF")
    )

    st.plotly_chart(
        corr_fig,
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "🧠 Model Performance":

    luxury_back_button()

    st.markdown(
        "<h1>🧠 Model Performance</h1>",
        unsafe_allow_html=True
    )

    best_accuracy = model_results[
        best_model_name
    ]["Accuracy"]

    luxury_card(
        f"""
        <div style="text-align:center;">

            <p style="
                color:#9CA5B3 !important;
                text-transform:uppercase;
                letter-spacing:0.08em;
            ">
                Best Performing Model
            </p>

            <h2 style="
                color:#F4D06F !important;
            ">
                🏆 {best_model_name}
            </h2>

            <h3 style="
                color:#FFFFFF !important;
            ">
                Test Accuracy:
                {best_accuracy:.2%}
            </h3>

        </div>
        """
    )

    rows = []

    for name, result in model_results.items():

        rows.append({
            "Model": name,
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1 Score": result["F1 Score"]
        })

    performance_df = pd.DataFrame(rows)

    st.dataframe(
        performance_df.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }
        ),
        use_container_width=True
    )

    fig = px.bar(
        performance_df,
        x="Model",
        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        barmode="group",
        title="Model Comparison"
    )

    fig.update_layout(
        paper_bgcolor="#0B0D12",
        plot_bgcolor="#0B0D12",
        font=dict(color="#FFFFFF")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "## 🔲 Confusion Matrix"
    )

    selected_model = st.selectbox(
        "Select Model",
        list(model_results.keys())
    )

    cm = model_results[
        selected_model
    ]["Confusion Matrix"]

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=target_names,
        y=target_names,
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count"
        },
        title=f"{selected_model} — Confusion Matrix"
    )

    cm_fig.update_layout(
        paper_bgcolor="#0B0D12",
        font=dict(color="#FFFFFF")
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif st.session_state.page == "🔬 Explainable AI":

    luxury_back_button()

    st.markdown(
        "<h1>🔬 Explainable AI</h1>",
        unsafe_allow_html=True
    )

    luxury_card(
        """
        <h2 style="
            color:#F4D06F !important;
        ">
            🌸 How IrisAI Thinks
        </h2>

        <p>
            IrisAI uses four numerical measurements
            to classify an Iris flower.
        </p>

        <ul>
            <li>Sepal Length</li>
            <li>Sepal Width</li>
            <li>Petal Length</li>
            <li>Petal Width</li>
        </ul>

        <p>
            These measurements are passed through the
            selected machine-learning model to predict
            one of the three Iris species.
        </p>
        """
    )

    st.markdown(
        "## 🌿 Average Measurements"
    )

    st.dataframe(
        species_means.style.format(
            "{:.2f}"
        ),
        use_container_width=True
    )

    st.markdown(
        "## 🌲 Random Forest Feature Importance"
    )

    rf = pipelines[
        "Random Forest"
    ].named_steps["clf"]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.feature_importances_
    }).sort_values(
        "Importance"
    )

    importance_fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    importance_fig.update_layout(
        paper_bgcolor="#0B0D12",
        plot_bgcolor="#0B0D12",
        font=dict(color="#FFFFFF")
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

    st.info(
        "Higher feature importance means the feature "
        "contributed more strongly to the Random Forest's "
        "decision-making process."
    )


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "📚 About Project":

    luxury_back_button()

    st.markdown(
        "<h1>📚 About IrisAI</h1>",
        unsafe_allow_html=True
    )

    luxury_card(
        f"""
        <h2 style="
            color:#F4D06F !important;
        ">
            🌸 IrisAI
        </h2>

        <p>
            IrisAI is a premium Streamlit machine-learning
            platform built around the classic Iris dataset.
        </p>

        <h3 style="
            color:#F4D06F !important;
        ">
            📊 Dataset
        </h3>

        <ul>
            <li>150 samples</li>
            <li>4 numerical features</li>
            <li>3 species/classes</li>
            <li>No missing values</li>
        </ul>

        <h3 style="
            color:#F4D06F !important;
        ">
            🤖 Machine Learning
        </h3>

        <ul>
            <li>Logistic Regression</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
            <li>K-Nearest Neighbors</li>
            <li>Support Vector Machine</li>
        </ul>

        <h3 style="
            color:#F4D06F !important;
        ">
            🛠️ Technologies
        </h3>

        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>Pandas</li>
            <li>NumPy</li>
            <li>Scikit-learn</li>
            <li>Plotly</li>
        </ul>

        <h3 style="
            color:#F4D06F !important;
        ">
            🏆 Best Model
        </h3>

        <p style="
            color:#FFFFFF !important;
            font-size:1.05rem;
        ">
            {best_model_name}
        </p>

        <p style="
            color:#F4D06F !important;
        ">
            Accuracy: {model_results[best_model_name]["Accuracy"]:.2%}
        </p>
        """
    )

    st.markdown(
        """
        <div class="footer">

            IrisAI • Premium Machine Learning Platform

            <br><br>

            Python • Streamlit • Scikit-learn • Plotly

        </div>
        """,
        unsafe_allow_html=True
    )
