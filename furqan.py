"""
IrisAI – Premium ML Classification Platform
A luxury, billionaire-tech style machine learning web application
built with Streamlit and the classic Iris dataset.
"""
import streamlit as st

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
    st.stop() # Password sahi nahi to app aage nahi chalegi
# PASSWORD WALA HISSA - Yahan khatam

# Tumhara purana code yahan se shuru hoga...
st.title("IrisAI - Premium ML Classification Platform")
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
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 114, 255, 0.4);
        background: linear-gradient(135deg, #00b4f0, #0066e6);
        color: white !important;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: white !important;
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

    # Define models inside pipelines (scaling only where beneficial)
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

    # Choose best model by accuracy
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

# Store best model in session state for easy access
if "best_model" not in st.session_state:
    st.session_state.best_model = best_pipeline
    st.session_state.best_model_name = best_model_name

# For explainable AI: species-wise feature means
@st.cache_data
def compute_species_means():
    return df.groupby("species_name")[feature_names].mean()

species_means = compute_species_means()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
    <h2 style="background: linear-gradient(135deg, #00F2FE, #4FACFE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent !important;">
        🌸 IrisAI
    </h2>
    <p style="color: #BBBBBB !important; font-size: 0.8rem; margin-top: -0.5rem;">Premium ML Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "📌 NAVIGATION",
    [
        "🏠 Dashboard",
        "🤖 AI Prediction",
        "📊 Dataset Explorer",
        "📈 Data Visualization",
        "🧠 Model Performance",
        "🔬 Explainable AI",
        "📚 About Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="footer" style="border-top: none; padding-top: 0.5rem;">
    <p style="color: #888888 !important;">Built with ❤️ using Streamlit</p>
    <p style="color: #888888 !important; font-size: 0.7rem;">© 2025 IrisAI Platform</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Helper: render a glass card
# -------------------------------
def glass_card(content, key=None):
    st.markdown(f'<div class="card">{content}</div>', unsafe_allow_html=True)

# -------------------------------
# 1. DASHBOARD
# -------------------------------
if page == "🏠 Dashboard":
    st.markdown("<h1 style='font-size: 2.5rem; color: #FFFFFF !important;'>🌸 Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #BBBBBB !important;'>Real‑time AI system overview</p>", unsafe_allow_html=True)

    # Status indicators
    col_status1, col_status2, col_status3 = st.columns([1, 1, 2])
    with col_status1:
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="status-dot"></span>
            <span style="font-weight: 500; color: #FFFFFF !important;">System Online</span>
        </div>
        """, unsafe_allow_html=True)
    with col_status2:
        st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <span class="status-dot" style="background-color: #00E676;"></span>
            <span style="font-weight: 500; color: #FFFFFF !important;">Model: {st.session_state.best_model_name}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_status3:
        st.markdown(f"""
        <div style="text-align: right; color: #BBBBBB !important;">
            Predictions made: <strong style="color: #FFFFFF !important;">{st.session_state.prediction_count}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # KPI Cards
    acc_best = model_results[best_model_name]["Accuracy"]
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
            <h3>4</h3>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>3</h3>
            <p>Classes</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{acc_best*100:.1f}%</h3>
            <p>Accuracy ({best_model_name})</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Dataset Health Report")
    glass_card(f"""
    <p style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Missing values:</strong> {df.isnull().sum().sum()}</p>
    <p style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Duplicated rows:</strong> {df.duplicated().sum()}</p>
    <p style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Data types:</strong> All numeric features + categorical target</p>
    <p style="color: #FFFFFF !important;"><strong style="color: #FFFFFF !important;">Class balance:</strong> Each species has 50 samples (perfectly balanced)</p>
    """)

    # Quick model table
    st.markdown("### 🧠 Model Performance Summary")
    metrics_df = pd.DataFrame(model_results).T.drop(columns=["Confusion Matrix"])
    st.dataframe(metrics_df.style.format("{:.2%}").highlight_max(axis=0, color="rgba(0,198,255,0.15)"), use_container_width=True)

# -------------------------------
# 2. AI PREDICTION
# -------------------------------
elif page == "🤖 AI Prediction":
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

            # Update session state
            st.session_state.prediction_count += 1
            st.session_state.last_prediction = {
                "species": pred_species,
                "confidence": confidence,
                "inputs": [sepal_len, sepal_wid, petal_len, petal_wid],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.history.append(st.session_state.last_prediction)

            st.markdown("---")
            # Result card
            glass_card(f"""
            <div style="text-align: center;">
                <h2 style="margin-bottom: 0.2rem; color: #FFFFFF !important;">🌸 {pred_species}</h2>
                <p style="font-size: 1.1rem; color: #4FACFE !important;">Confidence: {confidence*100:.1f}%</p>
            </div>
            """)

            # Probability bars
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

    # Show last prediction & history
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
elif page == "📊 Dataset Explorer":
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

    # Download dataset
    csv_full = df.to_csv(index=False).encode()
    st.download_button("📥 Download Full Dataset as CSV", csv_full, "iris_dataset.csv", "text/csv")

# -------------------------------
# 4. DATA VISUALIZATION
# -------------------------------
elif page == "📈 Data Visualization":
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
elif page == "🧠 Model Performance":
    st.markdown("<h1 style='color: #FFFFFF !important;'>🧠 Model Performance Analysis</h1>", unsafe_allow_html=True)

    # Metrics table
    metrics_df = pd.DataFrame(model_results).T.drop(columns=["Confusion Matrix"])
    st.markdown("### 📊 Metrics Summary")
    st.dataframe(metrics_df.style.format("{:.2%}").highlight_max(axis=0, color="rgba(0,198,255,0.2)"),
                 use_container_width=True)

    # Bar chart comparison
    st.markdown("### 📈 Accuracy Comparison")
    acc_series = metrics_df["Accuracy"]
    fig = px.bar(x=acc_series.index, y=acc_series.values, color=acc_series.index,
                 labels={"x": "Model", "y": "Accuracy"},
                 title="Model Accuracy Comparison")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix of best model
    st.markdown(f"### 🔍 Confusion Matrix – {best_model_name}")
    cm = model_results[best_model_name]["Confusion Matrix"]
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                       x=target_names, y=target_names, color_continuous_scale="Blues")
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="white")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Model details card
    glass_card(f"""
    <h4 style="color: #FFFFFF !important;">🏆 Best Model: {best_model_name}</h4>
    <p style="color: #FFFFFF !important;">Automatically selected based on highest test accuracy.</p>
    <p style="color: #FFFFFF !important;">Accuracy: <strong style="color: #FFFFFF !important;">{model_results[best_model_name]['Accuracy']:.2%}</strong></p>
    <p style="color: #FFFFFF !important;">Precision: {model_results[best_model_name]['Precision']:.2%} |
    Recall: {model_results[best_model_name]['Recall']:.2%} |
    F1: {model_results[best_model_name]['F1 Score']:.2%}</p>
    """)

# -------------------------------
# 6. EXPLAINABLE AI
# -------------------------------
elif page == "🔬 Explainable AI":
    st.markdown("<h1 style='color: #FFFFFF !important;'>🔬 Explainable AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #BBBBBB !important;'>Understand how the model reasons about your inputs.</p>", unsafe_allow_html=True)

    # Input from sidebar (same as prediction) – reuse the same form logic but standalone
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

        # Radar chart comparing input to species centroids
        categories = feature_names
        fig_radar = go.Figure()
        # Add input trace
        fig_radar.add_trace(go.Scatterpolar(
            r=input_vec.flatten(),
            theta=categories,
            fill='toself',
            name='Your Input',
            line=dict(color='#00F2FE', width=3)
        ))
        # Add species means
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

        # Euclidean distance to each centroid
        distances = {}
        for sp in target_names:
            means = species_means.loc[sp].values
            dist = np.linalg.norm(input_vec.flatten() - means)
            distances[sp] = dist
        dist_df = pd.DataFrame.from_dict(distances, orient='index', columns=['Distance'])
        dist_df['Similarity'] = 1 / (1 + dist_df['Distance'])
        st.markdown("### 📏 Distance to Species Centroids (Euclidean)")
        st.dataframe(dist_df.style.format("{:.3f}").highlight_min(subset=['Distance'], color="rgba(0,230,118,0.15)"))

        st.markdown("""
        <p style="color: #FFFFFF !important;">
        <strong style="color: #FFFFFF !important;">Interpretation:</strong> The species with the smallest distance to the input (and highest similarity) 
        is the most typical member of that class. The model's prediction often aligns with this nearest centroid, 
        but advanced models can capture more complex patterns.
        </p>
        """, unsafe_allow_html=True)

# -------------------------------
# 7. ABOUT PROJECT
# -------------------------------
elif page == "📚 About Project":
    st.markdown("<h1 style='color: #FFFFFF !important;'>📚 About the Project</h1>", unsafe_allow_html=True)

# -------------------------------
# 7. ABOUT PROJECT
# -------------------------------
elif page == "📚 About Project":
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
