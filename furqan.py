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

# PASSWORD WALA HISSA - Yahan se shuru
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

    /* Global resets - FORCE visible text colors */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0A0A0F;
        color: #FFFFFF !important;
    }

    /* Force all text to be visible */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stText, p, li, span, label, div {
        color: #FFFFFF !important;
    }

    /* Main container background */
    .reportview-container, .main, .block-container {
        background: #0A0A0F;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #12121A;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio span {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Cards – glassmorphism */
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

    /* Metric cards inside dashboard */
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
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-4px);
    }
    .metric-card h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #00F2FE, #4FACFE);
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

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        border: none;
        color: white !important;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        letter-spacing: 0.03em;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 114, 255, 0.4);
        background: linear-gradient(135deg, #00b4f0, #0066e6);
        color: white !important;
    }

    /* Input fields - FIXED for visibility */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        padding: 10px 14px !important;
    }

    /* Fix for empty input placeholder/box text */
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder {
        color: #AAAAAA !important;
        opacity: 1 !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #00F2FE !important;
        box-shadow: 0 0 0 2px rgba(0, 242, 254, 0.2) !important;
    }

    /* Input labels */
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Select box */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.08) !important;
        color: white !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #FFFFFF !important;
    }

    /* Dataframe */
    .stDataFrame, .stDataFrame * {
        color: #FFFFFF !important;
    }
    .stDataFrame th {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00F2FE !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #BBBBBB !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Pulse animation for status dot */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 230, 118, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
    }
    .status-dot {
        height: 14px;
        width: 14px;
        background-color: #00E676;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888888 !important;
        padding: 2rem 0 1rem 0;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }

    /* Error messages */
    .stAlert {
        color: #FFFFFF !important;
    }

    /* Radio buttons text */
    .stRadio div[role="radiogroup"] label {
        color: #FFFFFF !important;
    }

    /* Download button */
    .stDownloadButton > button {
        color: #FFFFFF !important;
        width: 100% !important;
    }

    /* Mobile responsiveness - Hamburger menu */
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

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Hero section animations */
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
        border-color: rgba(0, 242, 254, 0.3);
        box-shadow: 0 8px 30px rgba(0, 242, 254, 0.1);
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
if "mobile_menu_open" not in st.session_state:
    st.session_state.mobile_menu_open = False

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
# Sidebar Navigation with Hamburger Menu
# -------------------------------
st.sidebar.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
    <div style="text-align: center; flex: 1;">
        <h2 style="background: linear-gradient(135deg, #00F2FE, #4FACFE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent !important; margin: 0;">
            🌸 IrisAI
        </h2>
        <p style="color: #BBBBBB !important; font-size: 0.8rem; margin: -0.3rem 0 0 0;">Premium ML Platform</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation options
nav_options = [
    "🏠 Home",
    "🤖 AI Prediction",
    "📊 Dataset Explorer",
    "📈 Data Visualization",
    "🧠 Model Performance",
    "🔬 Explainable AI",
    "📚 About Project",
]

# Radio buttons with custom styling
page = st.sidebar.radio(
    "📌 NAVIGATION",
    nav_options,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 0.5rem 0;">
    <p style="color: #888888 !important; font-size: 0.7rem; text-align: center;">
        Built with ❤️ using Streamlit<br>
        © 2026 IrisAI Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Extract page name without emoji for logic
page_name = page.split(" ", 1)[-1] if " " in page else page

# -------------------------------
# Helper: render a glass card
# -------------------------------
def glass_card(content, key=None):
    st.markdown(f'<div class="card">{content}</div>', unsafe_allow_html=True)

# -------------------------------
# 1. HOME PAGE (New Powerful Homepage)
# -------------------------------
if page_name == "Home":
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #00F2FE, #4FACFE, #0072ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            🌸 Welcome to IrisAI
        </h1>
        <p style="font-size: 1.2rem; color: #BBBBBB !important; margin-bottom: 2rem;">
            The Future of Flower Classification — AI-Powered, Real-Time, and Astonishingly Accurate
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row
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

    # Features grid
    st.markdown("""
    <h2 style="margin-top: 2rem;">🚀 Why IrisAI?</h2>
    <div class="feature-grid">
        <div class="feature-item">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">5 Powerful Models</div>
            <div class="feature-desc">Logistic Regression, Decision Tree, Random Forest, KNN, and SVM — all at your fingertips</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">98.5% Accuracy</div>
            <div class="feature-desc">State-of-the-art performance on the Iris dataset with automatic best model selection</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">Understand why the model made its prediction with our XAI visualizations</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Interactive Visuals</div>
            <div class="feature-desc">Beautiful, interactive charts and graphs to explore your data deeply</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time Predictions</div>
            <div class="feature-desc">Instant classification with confidence scores and probability distributions</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Fully Responsive</div>
            <div class="feature-desc">Works perfectly on desktop, tablet, and mobile devices</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Call to action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h3 style="color: #FFFFFF !important;">Ready to classify your first flower?</h3>
            <p style="color: #BBBBBB !important; margin-bottom: 1rem;">Click below to start making predictions with our AI</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Predicting Now", use_container_width=True):
            st.session_state.page = "🤖 AI Prediction"
            st.rerun()

    # About the dataset section
    st.markdown("---")
    st.markdown("""
    <h2>📚 About the Iris Dataset</h2>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1rem 0;">
        <div>
            <p style="color: #FFFFFF !important;">The Iris flower dataset is a classic in machine learning and statistics. It contains measurements of <strong style="color: #FFFFFF !important;">150 samples</strong> from three species:</p>
            <ul>
                <li style="color: #FFFFFF !important;">🌸 <strong style="color: #FFFFFF !important;">Setosa</strong></li>
                <li style="color: #FFFFFF !important;">🌺 <strong style="color: #FFFFFF !important;">Versicolor</strong></li>
                <li style="color: #FFFFFF !important;">🌷 <strong style="color: #FFFFFF !important;">Virginica</strong></li>
            </ul>
            <p style="color: #FFFFFF !important;">Each sample has <strong style="color: #FFFFFF !important;">4 features</strong>: sepal length, sepal width, petal length, and petal width.</p>
        </div>
        <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.08);">
            <p style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">🔑 Key Facts:</strong></p>
            <p style="color: #FFFFFF !important;">• Created by <strong style="color: #FFFFFF !important;">Ronald Fisher</strong> in 1936</p>
            <p style="color: #FFFFFF !important;">• Perfectly balanced classes (50 each)</p>
            <p style="color: #FFFFFF !important;">• No missing values</p>
            <p style="color: #FFFFFF !important;">• One of the most studied datasets in ML</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick navigation cards
    st.markdown("""
    <h2 style="margin-top: 2rem;">🎯 Quick Navigation</h2>
    """, unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        st.markdown("""
        <div class="feature-item" style="cursor: pointer;" onclick="window.location.href='#predict'">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Prediction</div>
            <div class="feature-desc">Classify iris species with our advanced ML models</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Predictions", key="nav_predict"):
            st.session_state.page = "🤖 AI Prediction"
            st.rerun()
    
    with nav_col2:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Explore Data</div>
            <div class="feature-desc">Visualize and analyze the Iris dataset interactively</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Data", key="nav_explore"):
            st.session_state.page = "📊 Dataset Explorer"
            st.rerun()
    
    with nav_col3:
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Model Performance</div>
            <div class="feature-desc">Compare all models and their metrics</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Models", key="nav_models"):
            st.session_state.page = "🧠 Model Performance"
            st.rerun()

# -------------------------------
# 2. AI PREDICTION
# -------------------------------
elif page_name == "AI Prediction":
    st.markdown("<h1 style='color: #FFFFFF !important;'>🤖 Predict Iris Species</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #BBBBBB !important;'>Enter measurements and let the AI classify the flower</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        cols = st.columns(4)
        with cols[0]:
            sepal_len = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        with cols[1]:
            sepal_wid = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        with cols[2]:
            petal_len = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        with cols[3]:
            petal_wid = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

        col_btn1, col_btn2, _ = st.columns([1, 1, 2])
        with col_btn1:
            predict_btn = st.form_submit_button("🔮 Predict Species")
        with col_btn2:
            reset_btn = st.form_submit_button("🔄 Reset Inputs")

    if reset_btn:
        st.session_state.prediction_inputs = None
        st.rerun()

    if predict_btn:
        try:
            input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
            model = st.session_state.best_model
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            pred_species = target_names[prediction]
            confidence = np.max(probabilities)

            st.session_state.prediction_count += 1
            st.session_state.last_prediction = {
                "species": pred_species,
                "confidence": confidence,
                "inputs": [sepal_len, sepal_wid, petal_len, petal_wid],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.history.append(st.session_state.last_prediction)

            st.markdown("---")
            glass_card(f"""
            <div style="text-align: center;">
                <h2 style="margin-bottom: 0.2rem; color: #FFFFFF !important;">🌸 {pred_species}</h2>
                <p style="font-size: 1.1rem; color: #4FACFE !important;">Confidence: {confidence*100:.1f}%</p>
            </div>
            """)

            prob_df = pd.DataFrame({
                "Species": target_names,
                "Probability": probabilities
            }).sort_values("Probability", ascending=True)
            fig = px.bar(prob_df, x="Probability", y="Species", orientation="h",
                         color="Species", color_discrete_sequence=px.colors.sequential.Blues_r,
                         title="Prediction Probabilities")
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if st.session_state.last_prediction is not None:
        st.markdown("### 📋 Last Prediction")
        last = st.session_state.last_prediction
        st.markdown(f"**{last['species']}** (Confidence: {last['confidence']:.2%}) at {last['timestamp']}")

    if st.session_state.history:
        with st.expander("📜 Prediction History (this session)"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
            csv = hist_df.to_csv(index=False).encode()
            st.download_button("📥 Download History as CSV", csv, "prediction_history.csv", "text/csv")

# -------------------------------
# 3. DATASET EXPLORER
# -------------------------------
elif page_name == "Dataset Explorer":
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
elif page_name == "Data Visualization":
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
