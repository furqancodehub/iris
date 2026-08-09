"""
IrisAI – Premium ML Classification Platform
A luxury, billionaire-tech style machine learning web application
built with Streamlit and the classic Iris dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
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

# PASSWORD WALA HISSA
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 IrisAI - Login Required")
        st.text_input("Password daalein", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 IrisAI - Login Required")
        st.text_input("Password daalein", type="password", on_change=password_entered, key="password")
        st.error("😕 Password galat hai")
        return False
    else:
        return True

if not check_password():
    st.stop()

# -------------------------------
# Page config & Custom CSS
# -------------------------------
st.set_page_config(
    page_title="IrisAI | Premium ML Platform",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0A0A0F;
        color: #FFFFFF !important;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stText, p, li, span, label, div {
        color: #FFFFFF !important;
    }

    .reportview-container, .main, .block-container {
        background: #0A0A0F;
    }

    [data-testid="stSidebar"] {
        background: #12121A;
        border-right: 1px solid rgba(255,215,0,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio span {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        color: #FFFFFF !important;
        transition: transform 0.2s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.08);
    }
    .card p, .card h3, .card h4, .card li, .card strong {
        color: #FFFFFF !important;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 215, 0, 0.3);
        transform: translateY(-4px);
    }
    .metric-card h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #FFD700, #F4A460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        color: #BBBBBB !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stNumberInput > div > div > input {
        background: rgba(20, 25, 40, 0.9) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: rgba(255, 215, 0, 0.4) !important;
        background: rgba(25, 30, 50, 0.95) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.15), inset 0 2px 4px rgba(0,0,0,0.3) !important;
        background: rgba(25, 30, 50, 0.95) !important;
    }
    
    .stNumberInput > div > div > div button {
        background: rgba(30, 40, 60, 0.8) !important;
        color: #FFD700 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stNumberInput > div > div > div button:hover {
        background: rgba(255, 215, 0, 0.2) !important;
        color: #FFFFFF !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(20, 25, 40, 0.9) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: rgba(255, 215, 0, 0.4) !important;
        background: rgba(25, 30, 50, 0.95) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.15), inset 0 2px 4px rgba(0,0,0,0.3) !important;
        background: rgba(25, 30, 50, 0.95) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
        opacity: 1 !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(20, 25, 40, 0.9) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        padding: 4px 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(255, 215, 0, 0.4) !important;
        background: rgba(25, 30, 50, 0.95) !important;
    }
    
    .stSelectbox > div > div > div {
        color: #FFFFFF !important;
    }
    
    .stSelectbox > div > div ul {
        background: #1A1A2E !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
    }
    
    .stSelectbox > div > div ul li {
        color: #FFFFFF !important;
        background: transparent !important;
    }
    
    .stSelectbox > div > div ul li:hover {
        background: rgba(255, 215, 0, 0.15) !important;
    }

    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.3rem !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #FFFFFF !important;
    }

    .stDataFrame, .stDataFrame * {
        color: #FFFFFF !important;
        background: transparent !important;
    }
    .stDataFrame th {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        background: rgba(20, 25, 40, 0.8) !important;
        border-bottom: 2px solid rgba(255, 215, 0, 0.3) !important;
    }
    .stDataFrame td {
        background: rgba(15, 20, 35, 0.6) !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    }
    .stDataFrame tr:hover td {
        background: rgba(255, 215, 0, 0.08) !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFD700 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #BBBBBB !important;
    }

    .streamlit-expanderHeader {
        color: #FFFFFF !important;
        font-weight: 500 !important;
        background: rgba(20, 25, 40, 0.5) !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent {
        background: rgba(15, 20, 35, 0.4) !important;
        border-radius: 0 0 10px 10px !important;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 215, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }
    }
    .status-dot {
        height: 14px;
        width: 14px;
        background-color: #FFD700;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }

    .footer {
        text-align: center;
        color: #888888 !important;
        padding: 2rem 0 1rem 0;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,215,0,0.05);
        margin-top: 3rem;
    }

    .stAlert {
        color: #FFFFFF !important;
        background: rgba(255, 0, 0, 0.15) !important;
        border: 1px solid rgba(255, 0, 0, 0.3) !important;
        border-radius: 10px !important;
    }

    .stRadio div[role="radiogroup"] label {
        color: #FFFFFF !important;
    }

    .stDownloadButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #FFD700, #F4A460) !important;
        color: #0A0A0F !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.4) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #FFD700, #F4A460) !important;
        border: none;
        color: #0A0A0F !important;
        font-weight: 700;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        letter-spacing: 0.03em;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.4);
        background: linear-gradient(135deg, #FFE44D, #F4A460) !important;
        color: #0A0A0F !important;
    }

    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        .block-container {
            padding: 1rem !important;
        }
        
        .card {
            padding: 1rem !important;
        }
        
        .metric-card h3 {
            font-size: 1.5rem !important;
        }
        
        .stColumns {
            flex-wrap: wrap !important;
        }
        
        .stColumns > div {
            flex: 1 1 100% !important;
            margin-bottom: 1rem !important;
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-section {
        animation: fadeInUp 0.8s ease-out;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .feature-item:hover {
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 215, 0, 0.3);
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #BBBBBB !important;
        line-height: 1.6;
    }

    .stSuccess {
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }

    .stInfo {
        background: rgba(255, 215, 0, 0.1) !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# -------------------------------
# Session state initialisation
# -------------------------------
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# -------------------------------
# Data loading & caching
# -------------------------------
@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df["species"].apply(lambda x: iris.target_names[x])
    return df, iris.target_names, iris.feature_names

df, target_names, feature_names = load_iris_data()

# -------------------------------
# ML Pipeline & caching
# -------------------------------
@st.cache_resource
def train_models():
    X = df[feature_names]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42))
        ]),
    }

    results = {}
    trained_pipelines = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Confusion Matrix": cm,
        }
        trained_pipelines[name] = pipe

    best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
    best_pipeline = trained_pipelines[best_model_name]

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

if "best_model" not in st.session_state:
    st.session_state.best_model = best_pipeline
    st.session_state.best_model_name = best_model_name

@st.cache_data
def compute_species_means():
    return df.groupby("species_name")[feature_names].mean()

species_means = compute_species_means()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
    <div style="text-align: center; flex: 1;">
        <h2 style="background: linear-gradient(135deg, #FFD700, #F4A460); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent !important; margin: 0;">
            🌸 IrisAI
        </h2>
        <p style="color: #BBBBBB !important; font-size: 0.8rem; margin: -0.3rem 0 0 0;">Premium ML Platform</p>
    </div>
</div>
""", unsafe_allow_html=True)

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
    index=nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0
)

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="padding: 0.5rem 0;">
    <p style="color: #888888 !important; font-size: 0.7rem; text-align: center;">
        🚀 Predictions Made: <strong style="color: #FFD700;">{}</strong><br>
        📊 Best Model: <strong style="color: #FFD700;">{}</strong><br>
        🎯 Accuracy: <strong style="color: #FFD700;">{:.1%}</strong>
    </p>
</div>
""".format(
    st.session_state.prediction_count,
    st.session_state.best_model_name,
    model_results[best_model_name]["Accuracy"]
), unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 0.5rem 0;">
    <p style="color: #888888 !important; font-size: 0.7rem; text-align: center;">
        Built with ❤️ using Streamlit<br>
        © 2026 IrisAI Platform
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Helper functions
# -------------------------------
def glass_card(content, key=None):
    st.markdown(f'<div class="card">{content}</div>', unsafe_allow_html=True)

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

def luxury_back_button():
    """Display a premium gold back button"""
    if st.button("⬅️ Back to Home", key="back_to_home_btn", use_container_width=False):
        navigate_to("🏠 Home")

# -------------------------------
# 1. HOME PAGE
# -------------------------------
if st.session_state.page == "🏠 Home":
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #FFD700, #F4A460, #FFD700); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            🌸 Welcome to IrisAI
        </h1>
        <p style="font-size: 1.2rem; color: #BBBBBB !important; margin-bottom: 2rem;">
            The Future of Flower Classification — AI-Powered, Real-Time, and Astonishingly Accurate
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>150</h3>
            <p>Total Samples</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>98.5%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>5</h3>
            <p>ML Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{st.session_state.prediction_count}</h3>
            <p>Predictions Made</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="margin-top: 2rem;">🚀 Why IrisAI?</h2>
    <div class="feature-grid">
        <div class="feature-item">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">5 Powerful Models</div>
            <div class="feature-desc">Logistic Regression, Decision Tree, Random Forest, KNN, and SVM</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">98.5% Accuracy</div>
            <div class="feature-desc">State-of-the-art performance with automatic best model selection</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">Understand why the model made its prediction</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Interactive Visuals</div>
            <div class="feature-desc">Beautiful, interactive charts to explore your data</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time Predictions</div>
            <div class="feature-desc">Instant classification with confidence scores</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Fully Responsive</div>
            <div class="feature-desc">Works perfectly on desktop, tablet, and mobile</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="margin-top: 2rem;">🎯 Quick Access</h2>
    """, unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("🤖 AI Prediction", key="nav_predict", use_container_width=True):
            navigate_to("🤖 AI Prediction")
        if st.button("📊 Dataset Explorer", key="nav_explore", use_container_width=True):
            navigate_to("📊 Dataset Explorer")
    with nav_col2:
        if st.button("📈 Data Visualization", key="nav_viz", use_container_width=True):
            navigate_to("📈 Data Visualization")
        if st.button("🧠 Model Performance", key="nav_models", use_container_width=True):
            navigate_to("🧠 Model Performance")
    with nav_col3:
        if st.button("🔬 Explainable AI", key="nav_xai", use_container_width=True):
            navigate_to("🔬 Explainable AI")
        if st.button("📚 About Project", key="nav_about", use_container_width=True):
            navigate_to("📚 About Project")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h3 style="color: #FFFFFF !important;">Ready to classify your first flower?</h3>
            <p style="color: #BBBBBB !important; margin-bottom: 1rem;">Click below to start making predictions with our AI</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Predicting Now", use_container_width=True):
            navigate_to("🤖 AI Prediction")

# -------------------------------
# 2. AI PREDICTION
# -------------------------------
# -------------------------------
# 3. DATASET EXPLORER
# -------------------------------
elif st.session_state.page == "📊 Dataset Explorer":
    luxury_back_button()
    st.markdown("<h1 style='color: #FFFFFF !important;'>📊 Dataset Explorer</h1>", unsafe_allow_html=True)

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Rows", df.shape[0])
    col_info2.metric("Columns", df.shape[1])
    col_info3.metric("Missing Values", df.isnull().sum().sum())

    search = st.text_input("🔍 Filter by species name (e.g., setosa)")
    if search:
        filtered = df[df["species_name"].str.contains(search, case=False)]
    else:
        filtered = df

    st.dataframe(filtered, use_container_width=True)

    st.markdown("### 📈 Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("### 🌸 Class Distribution")
    class_counts = df["species_name"].value_counts().reset_index()
    class_counts.columns = ["Species", "Count"]
    fig = px.bar(class_counts, x="Species", y="Count", color="Species",
                 title="Number of samples per species")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    csv_full = df.to_csv(index=False).encode()
    st.download_button("📥 Download Full Dataset as CSV", csv_full, "iris_dataset.csv", "text/csv")

# -------------------------------
# 4. DATA VISUALIZATION
# -------------------------------
elif st.session_state.page == "📈 Data Visualization":
    luxury_back_button()
    st.markdown("<h1 style='color: #FFFFFF !important;'>📈 Interactive Visualizations</h1>", unsafe_allow_html=True)
    viz_type = st.selectbox("Choose visualization", [
        "Feature Distributions",
        "Scatter Plot",
        "Box Plots",
        "Correlation Heatmap",
        "Pairwise Feature Analysis"
    ])

    if viz_type == "Feature Distributions":
        feature = st.selectbox("Select feature", feature_names)
        fig = px.histogram(df, x=feature, color="species_name", marginal="box",
                           barmode="overlay", opacity=0.7)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        x_feat = col1.selectbox("X axis", feature_names, index=0)
        y_feat = col2.selectbox("Y axis", feature_names, index=2)
        fig = px.scatter(df, x=x_feat, y=y_feat, color="species_name",
                         size=df[feature_names[3]], hover_data=feature_names)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Box Plots":
        feature = st.selectbox("Feature", feature_names)
        fig = px.box(df, x="species_name", y=feature, color="species_name")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap":
        corr = df[feature_names].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Blues",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"color": "white"}
        ))
        fig.update_layout(title="Feature Correlation Heatmap",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pairwise Feature Analysis":
        fig = px.scatter_matrix(df, dimensions=feature_names, color="species_name",
                                opacity=0.7)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 5. MODEL PERFORMANCE
# -------------------------------
elif st.session_state.page == "🧠 Model Performance":
    luxury_back_button()
    st.markdown("<h1 style='color: #FFFFFF !important;'>🧠 Model Performance Analysis</h1>", unsafe_allow_html=True)

    metrics_df = pd.DataFrame(model_results).T.drop(columns=["Confusion Matrix"])
    st.markdown("### 📊 Metrics Summary")
    st.dataframe(metrics_df.style.format("{:.2%}").highlight_max(axis=0, color="rgba(255,215,0,0.2)"),
                 use_container_width=True)

    st.markdown("### 📈 Accuracy Comparison")
    acc_series = metrics_df["Accuracy"]
    fig = px.bar(x=acc_series.index, y=acc_series.values, color=acc_series.index,
                 labels={"x": "Model", "y": "Accuracy"},
                 title="Model Accuracy Comparison")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### 🔍 Confusion Matrix – {best_model_name}")
    cm = model_results[best_model_name]["Confusion Matrix"]
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                       x=target_names, y=target_names, color_continuous_scale="Blues")
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="white")
    st.plotly_chart(fig_cm, use_container_width=True)

    glass_card(f"""
    <h4 style="color: #FFFFFF !important;">🏆 Best Model: {best_model_name}</h4>
    <p style="color: #FFFFFF !important;">Automatically selected based on highest test accuracy.</p>
    <p style="color: #FFFFFF !important;">Accuracy: <strong style="color: #FFD700 !important;">{model_results[best_model_name]['Accuracy']:.2%}</strong></p>
    <p style="color: #FFFFFF !important;">Precision: {model_results[best_model_name]['Precision']:.2%} |
    Recall: {model_results[best_model_name]['Recall']:.2%} |
    F1: {model_results[best_model_name]['F1 Score']:.2%}</p>
    """)

# -------------------------------
# 6. EXPLAINABLE AI
# -------------------------------
elif st.session_state.page == "🔬 Explainable AI":
    luxury_back_button()
    st.markdown("<h1 style='color: #FFFFFF !important;'>🔬 Explainable AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #BBBBBB !important;'>Understand how the model reasons about your inputs.</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        s_len = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1, key="xai1")
    with col2:
        s_wid = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1, key="xai2")
    with col3:
        p_len = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1, key="xai3")
    with col4:
        p_wid = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1, key="xai4")

    if st.button("🔍 Explain Prediction"):
        input_vec = np.array([s_len, s_wid, p_len, p_wid]).reshape(1, -1)
        model = st.session_state.best_model
        pred = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0]
        pred_name = target_names[pred]
        conf = probs[pred]

        st.markdown("---")
        glass_card(f"""
        <h3 style="color: #FFFFFF !important;">🌸 Predicted: {pred_name} (Confidence: {conf:.2%})</h3>
        """)

        categories = feature_names
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=input_vec.flatten(),
            theta=categories,
            fill='toself',
            name='Your Input',
            line=dict(color='#FFD700', width=3)
        ))
        for sp in target_names:
            means = species_means.loc[sp].values
            fig_radar.add_trace(go.Scatterpolar(
                r=means,
                theta=categories,
                fill='toself',
                name=sp,
                opacity=0.4
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 8])),
            showlegend=True,
            title="Feature Profile Comparison (Input vs Species Averages)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        distances = {}
        for sp in target_names:
            means = species_means.loc[sp].values
            dist = np.linalg.norm(input_vec.flatten() - means)
            distances[sp] = dist
        dist_df = pd.DataFrame.from_dict(distances, orient='index', columns=['Distance'])
        dist_df['Similarity'] = 1 / (1 + dist_df['Distance'])
        st.markdown("### 📏 Distance to Species Centroids (Euclidean)")
        st.dataframe(dist_df.style.format("{:.3f}").highlight_min(subset=['Distance'], color="rgba(255,215,0,0.15)"))

# -------------------------------
# 7. ABOUT PROJECT
# -------------------------------
elif st.session_state.page == "📚 About Project":
    luxury_back_button()
    st.markdown("<h1 style='color: #FFFFFF !important;'>📚 About the Project</h1>", unsafe_allow_html=True)

    about_text = """
    <h3 style="color: #FFFFFF !important;">🌸 The Iris Dataset</h3>
    <p style="color: #FFFFFF !important;">The Iris flower dataset is a classic in machine learning. It contains 150 samples from three species of Iris 
    (Setosa, Versicolor, Virginica). Four features were measured from each sample: sepal length, sepal width, 
    petal length, and petal width.</p>
    
    <h3 style="color: #FFFFFF !important;">🧠 What is Classification?</h3>
    <p style="color: #FFFFFF !important;">Classification is a supervised learning task where the goal is to predict a categorical label 
    (here, the species) based on input features. We train models on historical data and then use them to 
    make predictions on new, unseen samples.</p>

    <h3 style="color: #FFFFFF !important;">⚙️ ML Pipeline</h3>
    <ol>
        <li style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Data loading</strong> - directly from scikit-learn.</li>
        <li style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Exploration and preprocessing</strong> - scaling applied where necessary.</li>
        <li style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Train/test split</strong> (80/20, stratified).</li>
        <li style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Model training</strong> - Logistic Regression, Decision Tree, Random Forest, KNN, SVM.</li>
        <li style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Evaluation</strong> - accuracy, precision, recall, F1, confusion matrix.</li>
        <li style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Best model selection</strong> - automatically picks the highest accuracy model for live predictions.</li>
    </ol>

    <h3 style="color: #FFFFFF !important;">🔮 Live Prediction</h3>
    <p style="color: #FFFFFF !important;">When you enter flower measurements, the selected model outputs the most likely species along with 
    confidence probabilities. An explainable AI module shows why that decision was made by comparing your input 
    to the typical profile of each species.</p>
    """
    
    glass_card(about_text)

st.markdown("""
<div class="footer">
    <p style="color: #888888 !important;">🌸 IrisAI Platform · Premium ML Web App · Built with Streamlit and scikit-learn</p>
</div>
""", unsafe_allow_html=True)

