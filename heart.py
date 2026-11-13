
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# PAGE CONFIG
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide"
)

# DESIGN
st.markdown("""
<style>
.stApp {
    background: #f7f5ff;
}
.heart {
    font-size: 65px;
    display: inline-block;
    animation: rotate 4s linear infinite;
}
@keyframes rotate {
    0% {transform: rotate(0deg);}
    100% {transform: rotate(360deg);}
}
.title {
    font-size: 34px;
    font-weight: bold;
    color: #8b3d91;
}
.subtitle {
    color: #777;
    font-size: 15px;
}
div[data-testid="stMetric"] {
    background: #1e1e2e;
color: white;
padding: 18px;
border-radius: 12px;
border-left: 4px solid #b35bba;
box-shadow: 0 2px 8px #333;
}
.stApp {
    background-color: #0e1117;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #161b2e;
}

div[data-testid="stMetric"] {
    background-color: #1e1e2e;
    color: white;
    border-left: 4px solid #b35bba;
    box-shadow: 0 2px 8px #333;
}
</style>
""", unsafe_allow_html=True)

# MODEL
@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# HEADER
c1, c2 = st.columns([1, 8])

with c1:
    st.markdown('<div class="heart">❤️</div>',
                unsafe_allow_html=True)

with c2:
    st.markdown(
        '<div class="title">Heart Disease Prediction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Smart Healthcare Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

st.divider()

# SIDEBAR
st.sidebar.title("❤️ Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    ["Overview", "Patient Analytics", "Prediction", "Model Insights"]
)

st.sidebar.info("Heart Disease Analytics Project")

# DATASET
features = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

df = None

for filename in ["heart.csv", "Heart.csv", "heart_disease.csv"]:
    if os.path.exists(filename):
        try:
            temp = pd.read_csv(filename)
            if set(features + ["target"]).issubset(temp.columns):
                df = temp
                break
        except Exception:
            pass

if df is None:
    uploaded = st.sidebar.file_uploader(
        "Upload Heart CSV (optional)",
        type=["csv"]
    )
    if uploaded is not None:
        temp = pd.read_csv(uploaded)
        if set(features + ["target"]).issubset(temp.columns):
            df = temp
        else:
            st.sidebar.error("CSV columns do not match.")

# SESSION HISTORY
if "history" not in st.session_state:
    st.session_state.history = []

# OVERVIEW
if page == "Overview":

    st.subheader("📊 Project Overview")

    if df is not None:
        total = len(df)
        disease = int((df["target"] == 1).sum())
        healthy = int((df["target"] == 0).sum())
        avg_age = df["age"].mean()

        a, b, c, d = st.columns(4)

        a.metric("Total Patients", total)
        b.metric("Heart Disease", disease)
        c.metric("No Disease", healthy)
        d.metric("Average Age", f"{avg_age:.1f}")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown("**Heart Disease Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(
                data=df, x="target",
                hue="target",
                palette=["#6385d8", "#d45b99"],
                legend=False, ax=ax
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No Disease", "Disease"])
            ax.set_xlabel("")
            ax.set_ylabel("Patients")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("**Age Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(
                df["age"], kde=True,
                color="#a45bc4", ax=ax
            )
            ax.set_xlabel("Age")
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("Upload your heart CSV from the sidebar to view patient analytics.")
        st.metric("Model Status", "Loaded Successfully")
        st.write("Your prediction model is ready.")

    st.divider()
    st.markdown("### ❤️ About This Project")
    st.write(
        "A Machine Learning project that analyzes patient "
        "health information and predicts the possibility of "
        "heart disease using a trained Random Forest model."
    )

# PATIENT ANALYTICS
elif page == "Patient Analytics":

    st.subheader("📈 Patient Analytics")

    if df is not None:

        left, right = st.columns(2)

        with left:
            st.markdown("**Gender Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(
                data=df, x="sex",
                hue="sex",
                palette=["#d45b99", "#648be0"],
                legend=False, ax=ax
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Female", "Male"])
            ax.set_xlabel("Gender")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("**Chest Pain Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(
                data=df, x="cp",
                hue="cp",
                palette="viridis",
                legend=False, ax=ax
            )
            ax.set_xlabel("Chest Pain Type")
            st.pyplot(fig)
            plt.close(fig)

        left, right = st.columns(2)

        with left:
            st.markdown("**Cholesterol Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(
                df["chol"], kde=True,
                color="#9b59b6", ax=ax
            )
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("**Blood Pressure Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(
                df["trestbps"], kde=True,
                color="#3989ce", ax=ax
            )
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Feature Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            df.corr(numeric_only=True),
            cmap="coolwarm",
            annot=True,
            fmt=".1f",
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("Upload the heart CSV to see analytics.")

# PREDICTION
elif page == "Prediction":

    st.subheader("❤️ Heart Disease Prediction")
    st.write("Enter patient health details below.")

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 18, 100, 45)
            sex = st.selectbox("Gender", [0, 1],
                               format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input("Blood Pressure", 80, 250, 130)
            chol = st.number_input("Cholesterol", 100, 600, 240)

        with c2:
            fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])

        with c3:
            oldpeak = st.number_input(
                "Oldpeak", 0.0, 10.0, 1.0, step=0.1
            )
            slope = st.selectbox("Slope", [0, 1, 2])
            ca = st.selectbox("Major Vessels (CA)", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thal", [0, 1, 2, 3])

        submitted = st.form_submit_button(
            "❤️ Predict Heart Disease",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs,
              restecg, thalach, exang, oldpeak,
              slope, ca, thal]],
            columns=features
        )

        try:
            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = float(max(probabilities)) * 100
            else:
                confidence = None

            if prediction == 1:
                st.error("❤️ Prediction: Heart Disease Detected")
            else:
                st.success("💚 Prediction: No Heart Disease Detected")

            if confidence is not None:
                st.metric("Model Confidence", f"{confidence:.2f}%")

            st.warning(
                "This is an educational ML prediction, not a medical diagnosis. "
                "Consult a qualified doctor for medical advice."
            )

            st.session_state.history.append({
                "Age": age,
                "Gender": sex,
                "Prediction": int(prediction),
                "Confidence": confidence
            })

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# MODEL INSIGHTS
elif page == "Model Insights":

    st.subheader("🤖 Model Insights")

    st.success("Random Forest Model Loaded")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.markdown("**Important Features**")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=importance,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="Purples",
            legend=False,
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Model Information**")
    st.write("Algorithm: Random Forest Classifier")
    st.write("Number of Features: 13")
    st.write("Model File: heart_model.pkl")

    st.info("ROC-AUC achieved during notebook evaluation: 0.86")

# PREDICTION HISTORY
st.divider()

st.subheader("📋 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction Report",
        data=csv,
        file_name="heart_prediction_history.csv",
        mime="text/csv"
    )

else:
    st.caption("No predictions made yet.")

# FOOTER
st.divider()

st.markdown(
    "<center>❤️ Heart Disease Prediction | Machine Learning Project</center>",
    unsafe_allow_html=True
) 
