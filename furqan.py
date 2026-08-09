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
        st.title("🔒 IrisAI - Login Required")

        st.text_input(
            "Password daalein",
            type="password",
            on_change=password_entered,
            key="password",
        )

        return False

    elif not st.session_state["password_correct"]:
        st.title("🔒 IrisAI - Login Required")

        st.text_input(
            "Password daalein",
            type="password",
            on_change=password_entered,
            key="password",
        )

        st.error("😕 Password galat hai")

        return False

    else:
        return True


if not check_password():
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

def load_css():

    st.markdown(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0A0A0F;
            color: #FFFFFF !important;
        }

        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown span,
        .stText,
        p,
        li,
        span,
        label,
        div {
            color: #FFFFFF !important;
        }

        .main,
        .block-container {
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
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
            color: #FFFFFF !important;
            transition: transform 0.2s ease;
        }

        .card:hover {
            transform: translateY(-4px);
            background: rgba(255, 255, 255, 0.08);
        }

        .card p,
        .card h3,
        .card h4,
        .card li,
        .card strong {
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
        }

        .stNumberInput > div > div > input:hover {
            border-color: rgba(255, 215, 0, 0.4) !important;
            background: rgba(25, 30, 50, 0.95) !important;
        }

        .stNumberInput > div > div > input:focus {
            border-color: #FFD700 !important;
            box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.15) !important;
            background: rgba(25, 30, 50, 0.95) !important;
        }

        .stNumberInput > div > div > div button {
            background: rgba(30, 40, 60, 0.8) !important;
            color: #FFD700 !important;
            border: none !important;
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
        }

        .stTextInput > div > div > input:hover {
            border-color: rgba(255, 215, 0, 0.4) !important;
            background: rgba(25, 30, 50, 0.95) !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #FFD700 !important;
            box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.15) !important;
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
        }

        .stSelectbox > div > div:hover {
            border-color: rgba(255, 215, 0, 0.4) !important;
            background: rgba(25, 30, 50, 0.95) !important;
        }

        .stSelectbox > div > div > div {
            color: #FFFFFF !important;
        }

        .stNumberInput label,
        .stTextInput label,
        .stSelectbox label {
            color: rgba(255, 255, 255, 0.8) !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: #FFFFFF !important;
        }

        [data-testid="stMetricValue"] {
            color: #FFD700 !important;
            font-weight: 700 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #BBBBBB !important;
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
        }

        .stButton > button {
            background: linear-gradient(135deg, #FFD700, #F4A460) !important;
            border: none !important;
            color: #0A0A0F !important;
            font-weight: 700 !important;
            padding: 0.6rem 2rem !important;
            border-radius: 10px !important;
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(255, 215, 0, 0.4);
            color: #0A0A0F !important;
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

        @media (max-width: 768px) {

            .block-container {
                padding: 1rem !important;
            }

            .card {
                padding: 1rem !important;
            }

            .metric-card h3 {
                font-size: 1.5rem !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "history" not in st.session_state:
    st.session_state.history = []

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_iris_data():

    iris = load_iris()

    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )

    df["species"] = iris.target

    df["species_name"] = df["species"].apply(
        lambda x: iris.target_names[x]
    )

    return df, iris.target_names, iris.feature_names


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

        acc = accuracy_score(
            y_test,
            y_pred
        )

        prec = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        rec = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Confusion Matrix": cm,
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
    <div style="text-align:center;">
        <h1>🌸 IrisAI</h1>
        <p style="color:#BBBBBB !important;">
            Premium ML Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

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
    <div style="text-align:center;">
        <p style="color:#BBBBBB !important;">
            Best Model
        </p>
        <h3 style="color:#FFD700 !important;">
            {best_model_name}
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def glass_card(content, key=None):
    st.markdown(
        f'<div class="card">{content}</div>',
        unsafe_allow_html=True
    )


def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def luxury_back_button():

    if st.button(
        "⬅️ Back to Home",
        key="back_to_home_btn",
        use_container_width=False
    ):
        navigate_to("🏠 Home")


# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == "🏠 Home":

    st.markdown(
        """
        <div class="hero-section"
             style="text-align:center; padding:2rem 0;">

            <h1 style="font-size:3.5rem;">
                🌸 Welcome to IrisAI
            </h1>

            <p style="
                color:#BBBBBB !important;
                font-size:1.2rem;
            ">
                The Future of Flower Classification —
                AI-Powered, Real-Time, and Astonishingly Accurate
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    current_accuracy = model_results[
        best_model_name
    ]["Accuracy"]

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
                <p>Best Model Accuracy</p>
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
                <p>Predictions Made</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <h2 style="margin-top:2rem;">
            🚀 Why IrisAI?
        </h2>

        <div class="feature-grid">

            <div class="feature-item">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">
                    5 Powerful Models
                </div>
                <div class="feature-desc">
                    Logistic Regression, Decision Tree,
                    Random Forest, KNN, and SVM
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">
                    High Accuracy
                </div>
                <div class="feature-desc">
                    Automatic best-model selection
                    based on test accuracy
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">🔬</div>
                <div class="feature-title">
                    Explainable AI
                </div>
                <div class="feature-desc">
                    Understand the model's predictions
                    and important features
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-title">
                    Interactive Visuals
                </div>
                <div class="feature-desc">
                    Beautiful interactive charts
                    for exploring the Iris dataset
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">
                    Real-Time Predictions
                </div>
                <div class="feature-desc">
                    Instant flower classification
                    with probability scores
                </div>
            </div>

            <div class="feature-item">
                <div class="feature-icon">📱</div>
                <div class="feature-title">
                    Responsive Interface
                </div>
                <div class="feature-desc">
                    Designed for desktop,
                    tablet, and mobile
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style="margin-top:2rem;">
            🎯 Quick Access
        </h2>
        """,
        unsafe_allow_html=True
    )

    col_nav1, col_nav2, col_nav3 = st.columns(3)

    with col_nav1:

        if st.button(
            "🤖 AI Prediction",
            key="nav_predict",
            use_container_width=True
        ):
            navigate_to("🤖 AI Prediction")

        if st.button(
            "📊 Dataset Explorer",
            key="nav_explore",
            use_container_width=True
        ):
            navigate_to("📊 Dataset Explorer")

    with col_nav2:

        if st.button(
            "📈 Data Visualization",
            key="nav_viz",
            use_container_width=True
        ):
            navigate_to("📈 Data Visualization")

        if st.button(
            "🧠 Model Performance",
            key="nav_models",
            use_container_width=True
        ):
            navigate_to("🧠 Model Performance")

    with col_nav3:

        if st.button(
            "🔬 Explainable AI",
            key="nav_xai",
            use_container_width=True
        ):
            navigate_to("🔬 Explainable AI")

        if st.button(
            "📚 About Project",
            key="nav_about",
            use_container_width=True
        ):
            navigate_to("📚 About Project")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:2rem 0;
            ">

                <h3>
                    Ready to classify your first flower?
                </h3>

                <p style="
                    color:#BBBBBB !important;
                ">
                    Click below to start making predictions
                    with our AI
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🚀 Start Predicting Now",
            use_container_width=True
        ):
            navigate_to("🤖 AI Prediction")


# ============================================================
# PAGE 2 — AI PREDICTION
# ============================================================

elif st.session_state.page == "🤖 AI Prediction":

    luxury_back_button()

    st.markdown(
        """
        <h1>
            🤖 Predict Iris Species
        </h1>

        <p style="color:#BBBBBB !important;">
            Enter measurements and let the AI classify the flower.
        </p>
        """,
        unsafe_allow_html=True
    )

    with st.form(key="prediction_form"):

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
            "🔮 Predict Species"
        )

    if submitted:

        try:

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

            pred_species = target_names[prediction]

            confidence = np.max(probabilities)

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            prediction_record = {
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

            st.session_state.last_prediction = (
                prediction_record
            )

            st.session_state.history.append(
                prediction_record
            )

            glass_card(
                f"""
                <div style="text-align:center;">

                    <h2>
                        🌸 {pred_species}
                    </h2>

                    <p style="
                        font-size:1.1rem;
                        color:#FFD700 !important;
                    ">
                        Confidence:
                        {confidence * 100:.1f}%
                    </p>

                </div>
                """
            )

            prob_df = pd.DataFrame({
                "Species": target_names,
                "Probability": probabilities
            }).sort_values(
                "Probability",
                ascending=True
            )

            fig = px.bar(
                prob_df,
                x="Probability",
                y="Species",
                orientation="h",
                color="Species",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                title="Prediction Probabilities"
            )

            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {str(e)}"
            )

    if st.session_state.last_prediction is not None:

        st.markdown("### 📋 Last Prediction")

        last = st.session_state.last_prediction

        st.info(
            f"🌸 **{last['species']}** "
            f"(Confidence: {last['confidence']:.2%}) "
            f"at {last['timestamp']}"
        )

    if st.session_state.history:

        with st.expander(
            "📜 Prediction History (this session)"
        ):

            hist_df = pd.DataFrame(
                st.session_state.history
            )

            st.dataframe(
                hist_df,
                use_container_width=True
            )

            csv = hist_df.to_csv(
                index=False
            ).encode()

            st.download_button(
                "📥 Download History as CSV",
                csv,
                "prediction_history.csv",
                "text/csv"
            )


# ============================================================
# PAGE 3 — DATASET EXPLORER
# ============================================================

elif st.session_state.page == "📊 Dataset Explorer":

    luxury_back_button()

    st.markdown(
        "<h1>📊 Dataset Explorer</h1>",
        unsafe_allow_html=True
    )

    col_info1, col_info2, col_info3 = st.columns(3)

    col_info1.metric(
        "Rows",
        df.shape[0]
    )

    col_info2.metric(
        "Columns",
        df.shape[1]
    )

    col_info3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    search = st.text_input(
        "🔍 Filter by species name (e.g., setosa)"
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
        "### 📈 Descriptive Statistics"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown(
        "### 🌸 Class Distribution"
    )

    class_counts = (
        df["species_name"]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = [
        "Species",
        "Count"
    ]

    fig = px.bar(
        class_counts,
        x="Species",
        y="Count",
        color="Species",
        title="Number of Samples per Species"
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 4 — DATA VISUALIZATION
# ============================================================

elif st.session_state.page == "📈 Data Visualization":

    luxury_back_button()

    st.markdown(
        "<h1>📈 Data Visualization</h1>",
        unsafe_allow_html=True
    )

    selected_x = st.selectbox(
        "X-axis Feature",
        feature_names,
        index=0
    )

    selected_y = st.selectbox(
        "Y-axis Feature",
        feature_names,
        index=2
    )

    fig = px.scatter(
        df,
        x=selected_x,
        y=selected_y,
        color="species_name",
        title=f"{selected_x} vs {selected_y}",
        hover_data=feature_names
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### 📊 Feature Distributions")

    selected_feature = st.selectbox(
        "Select Feature",
        feature_names,
        key="distribution_feature"
    )

    hist_fig = px.histogram(
        df,
        x=selected_feature,
        color="species_name",
        marginal="box",
        title=f"Distribution of {selected_feature}"
    )

    hist_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        hist_fig,
        use_container_width=True
    )

    st.markdown("### 🔥 Feature Correlation")

    correlation = df[
        feature_names
    ].corr()

    corr_fig = px.imshow(
        correlation,
        text_auto=True,
        title="Feature Correlation Matrix"
    )

    corr_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        corr_fig,
        use_container_width=True
    )


# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "🧠 Model Performance":

    luxury_back_button()

    st.markdown(
        "<h1>🧠 Model Performance</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">

            <h3>🏆 Best Model</h3>

            <h2 style="color:#FFD700 !important;">
                {best_model_name}
            </h2>

            <p>
                Selected using the highest test-set accuracy.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    performance_rows = []

    for name, result in model_results.items():

        performance_rows.append({
            "Model": name,
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1 Score": result["F1 Score"],
        })

    performance_df = pd.DataFrame(
        performance_rows
    )

    st.dataframe(
        performance_df.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
            }
        ),
        use_container_width=True
    )

    metric_fig = px.bar(
        performance_df,
        x="Model",
        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        barmode="group",
        title="Model Performance Comparison"
    )

    metric_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        metric_fig,
        use_container_width=True
    )

    st.markdown("### 🔲 Confusion Matrix")

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
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True
    )


# ============================================================
# PAGE 6 — EXPLAINABLE AI
# ============================================================

elif st.session_state.page == "🔬 Explainable AI":

    luxury_back_button()

    st.markdown(
        "<h1>🔬 Explainable AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

            <h3>🌸 How does IrisAI classify flowers?</h3>

            <p>
                IrisAI uses four measurements:
            </p>

            <ul>
                <li>Sepal Length</li>
                <li>Sepal Width</li>
                <li>Petal Length</li>
                <li>Petal Width</li>
            </ul>

            <p>
                These measurements are provided to the selected
                machine-learning model, which predicts one of
                three Iris species.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🌿 Average Measurements by Species"
    )

    st.dataframe(
        species_means.style.format("{:.2f}"),
        use_container_width=True
    )

    st.markdown(
        "### 🌸 Feature Importance — Random Forest"
    )

    rf_pipeline = pipelines[
        "Random Forest"
    ]

    rf_model = rf_pipeline.named_steps["clf"]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    importance_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

    st.info(
        "Feature importance shows how useful each feature "
        "was for the Random Forest model when making decisions."
    )


# ============================================================
# PAGE 7 — ABOUT PROJECT
# ============================================================

elif st.session_state.page == "📚 About Project":

    luxury_back_button()

    st.markdown(
        "<h1>📚 About IrisAI</h1>",
        unsafe_allow_html=True
    )

    glass_card(
        """
        <h2>🌸 IrisAI</h2>

        <p>
            IrisAI is a Streamlit-based machine-learning
            classification platform built around the classic
            Iris dataset.
        </p>

        <h3>📊 Dataset</h3>

        <ul>
            <li>150 total samples</li>
            <li>4 numerical features</li>
            <li>3 Iris species</li>
            <li>No missing values</li>
        </ul>

        <h3>🤖 Machine Learning Models</h3>

        <ul>
            <li>Logistic Regression</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
            <li>K-Nearest Neighbors</li>
            <li>Support Vector Machine</li>
        </ul>

        <h3>🛠️ Technologies</h3>

        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>Pandas</li>
            <li>NumPy</li>
            <li>Scikit-learn</li>
            <li>Plotly</li>
        </ul>
        """
    )

    st.markdown(
        "### 🏆 Current Best Model"
    )

    st.success(
        f"{best_model_name} — "
        f"{model_results[best_model_name]['Accuracy']:.2%} "
        f"test accuracy"
    )

    st.markdown(
        """
        <div class="footer">
            IrisAI • Premium Machine Learning Platform
            <br>
            Built with Python + Streamlit + Scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )
