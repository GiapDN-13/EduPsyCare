"""
EduPsyCare Dashboard - Refined Prediction, EDA, and Model Evaluation
Objective:
Build a comprehensive, interactive dashboard using Streamlit for the EduPsyCare project. The dashboard provides three primary functions:
    1. Prediction Tab:
       - Allow users to input individual feature values.
       - Dynamically compute and display a consolidated "Pressure Index" based on academic pressure, financial stress, and work pressure.
       - Generate model predictions along with their probabilities, ensuring that the computed Pressure Index used in predictions is consistent with what is displayed.
    2. EDA & Feature Relationships Tab:
       - Enable users to explore data distributions through interactive, side-by-side plots (histogram and boxplot) for any chosen numeric feature.
       - Provide a full-width, large-scale correlation heatmap for deep analysis of relationships among numeric features.
       - Offer an interactive scatter matrix (or alternative relationship plots) to visualize inter-feature dependencies and patterns.
    3. Model Evaluation Tab:
       - Present key performance metrics visually using an interactive radar chart that compares F1 Score, Accuracy, Precision, and Recall across multiple models.
       - Include additional evaluation plots such as F1 Score bar chart, confusion matrix, ROC curve, and precision-recall curve for detailed model performance analysis.
By integrating these functionalities, the dashboard delivers clear, actionable insights that support data-driven decision-making in predicting student depression risk and guiding timely interventions.

Author: Dang Nguyen Giap
Date: 19-03-2025
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve)

# ------------------------------
# Global Constants
# ------------------------------
BAR_COLOR = "#636EFA"
LINE_COLOR = "#FF6600"
LINE_WIDTH = 2

# ------------------------------
# Global Setup and Sidebar
# ------------------------------
st.set_page_config(page_title="EduPsyCare Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("EduPsyCare Model Dashboard")

st.sidebar.header("Dashboard Information")
st.sidebar.info("This dashboard provides Prediction, EDA, and Model Evaluation insights for the EduPsyCare project.")

# Attempt to load the best model
try:
    best_model = joblib.load("../models/depression_model.pkl")
    st.sidebar.success("Best model loaded successfully.")
except Exception as e:
    st.sidebar.error("Error loading best model: " + str(e))
    best_model = None

# Simulated results for demonstration
results = {
    "LogisticRegression": {"F1 Score": 0.72, "Accuracy": 0.70, "Precision": 0.68, "Recall": 0.75},
    "RandomForest": {"F1 Score": 0.75, "Accuracy": 0.74, "Precision": 0.73, "Recall": 0.76},
    "XGBoost": {"F1 Score": 0.78, "Accuracy": 0.77, "Precision": 0.78, "Recall": 0.77},
    "SVM": {"F1 Score": 0.70, "Accuracy": 0.69, "Precision": 0.67, "Recall": 0.72},
    "ExtraTrees": {"F1 Score": 0.74, "Accuracy": 0.73, "Precision": 0.72, "Recall": 0.74}
}
best_model_name = max(results, key=lambda name: results[name]["F1 Score"])
st.sidebar.markdown(f"**Best Model:** {best_model_name}")

# Load dataset for EDA
try:
    df = pd.read_csv("../data/Student_Depression_Dataset_clean.csv")
    st.sidebar.success("Dataset loaded for EDA.")
except Exception as e:
    st.sidebar.error("Error loading dataset for EDA: " + str(e))
    df = None

# ------------------------------
# Create Tabs: Prediction, EDA, Model Evaluation
# ------------------------------
tabs = st.tabs(["Prediction", "EDA & Feature Relationships", "Model Evaluation"])

# ------------------------------------------------
# Tab 1: Prediction
# ------------------------------------------------

with tabs[0]:
   st.header("Prediction")

   col_input, col_summary = st.columns(2)

   with col_input:
      st.subheader("Enter Feature Values")
      age = st.number_input("Age", min_value=15, max_value=100, value=25)
      cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
      work_hours = st.number_input("Work/Study Hours", min_value=0.0, value=5.0, step=0.5)
      sleep_hours = st.number_input("Sleep Duration (hours)", min_value=0.0, value=6.0, step=0.5)
      academic_pressure = st.number_input("Academic Pressure", min_value=0.0, value=3.0, step=1.0)
      financial_stress = st.number_input("Financial Stress", min_value=0.0, value=2.0, step=1.0)
      work_pressure = st.number_input("Work Pressure", min_value=0.0, value=1.0, step=1.0)
      gender = st.selectbox("Gender", options=["Male", "Female"])
      family_history = st.selectbox("Family History of Mental Illness", options=["No", "Yes"])

   with col_summary:
      st.subheader("Input Summary")
      st.write("**Age:**", age)
      st.write("**CGPA:**", cgpa)
      st.write("**Work/Study Hours:**", work_hours)
      st.write("**Sleep Duration:**", sleep_hours)
      st.write("**Gender:**", gender)
      st.write("**Family History:**", family_history)
      st.write("**Academic Pressure:**", academic_pressure)
      st.write("**Financial Stress:**", financial_stress)
      st.write("**Work Pressure:**", work_pressure)

   # Predict button
   if st.button("Predict"):
      # Compute final Pressure Index once
      pressure_index = float(academic_pressure) + float(financial_stress) + float(work_pressure)

      # Encode categorical variables (Male=1, Female=0; Yes=1, No=0)
      gender_encoded = 1 if gender == "Male" else 0
      family_encoded = 1 if family_history == "Yes" else 0

      # Create a sample DataFrame for prediction
      sample = pd.DataFrame({
         "Age": [age],
         "CGPA": [cgpa],
         "Work/Study Hours": [work_hours],
         "Sleep Duration Numeric": [sleep_hours],
         "Gender_encoded": [gender_encoded],
         "FamilyHistory_encoded": [family_encoded],
         "Academic Pressure": [academic_pressure],
         "Financial Stress": [financial_stress],
         "Work Pressure": [work_pressure],
         "pressure_index": [pressure_index]
      })

      if best_model is not None:
         prediction = best_model.predict(sample)[0]
         pred_text = "Yes" if prediction == 1 else "No"
         if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(sample)[0, 1]
         else:
            proba = None

         st.subheader("Prediction Result")
         st.write("Predicted Depression:", pred_text)
         if proba is not None:
            st.write("Prediction Probability (Depression):", round(proba, 4))

         # Create two columns for the two gauges (Pressure Index and Prediction Probability)
         col_gauge1, col_gauge2 = st.columns(2)

         with col_gauge1:
            st.write("**Pressure Index Gauge**")
            fig_pressure = go.Figure(go.Indicator(
               mode="gauge+number",
               value=pressure_index,
               title={"text": "Pressure Index"},
               gauge={"axis": {"range": [0, 50]}, "bar": {"color": "red"}}
            ))
            st.plotly_chart(fig_pressure, use_container_width=True)

         with col_gauge2:
            if proba is not None:
               st.write("**Depression Probability Gauge**")
               fig_prob = go.Figure(go.Indicator(
                  mode="gauge+number",
                  value=proba * 100,
                  title={"text": "Depression Probability (%)"},
                  gauge={"axis": {"range": [0, 100]}, "bar": {"color": "blue"}}
               ))
               st.plotly_chart(fig_prob, use_container_width=True)
            else:
               st.write("Probability gauge not available for this model.")
      else:
         st.error("Best model is not available.")

# ------------------------------------------------
# Tab 2: EDA & Feature Relationships
# ------------------------------------------------
with tabs[1]:
    st.header("EDA & Feature Relationships")
    if df is None:
        st.error("No dataset available for EDA.")
    else:
        st.subheader("Data Distribution")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        chosen_col = st.selectbox("Select a numeric column for distribution", numeric_cols, key="dist_col")

        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            fig_hist = px.histogram(df, x=chosen_col, nbins=30,
                                    title=f"Histogram of {chosen_col}",
                                    color_discrete_sequence=[BAR_COLOR])
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_dist2:
            fig_box = px.box(df, y=chosen_col,
                             title=f"Boxplot of {chosen_col}",
                             color_discrete_sequence=[BAR_COLOR])
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             title="Correlation Heatmap", height=1000)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Scatter Matrix")
        subset_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        fig_scatter = px.scatter_matrix(df, dimensions=subset_cols,
                                        title="Scatter Matrix of Numeric Features")
        fig_scatter.update_layout(height=800)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------
# Tab 3: Model Evaluation
# ------------------------------------------------
with tabs[2]:
    st.header("Model Evaluation")

    # Radar chart comparing F1 Score, Accuracy, Precision, Recall
    st.subheader("Radar Chart: Key Metrics")
    categories = ["F1 Score", "Accuracy", "Precision", "Recall"]
    radar_data = []
    for model_name in results:
        metrics = [results[model_name][cat] for cat in categories]
        radar_data.append(go.Scatterpolar(
            r=metrics,
            theta=categories,
            fill='toself',
            name=model_name
        ))
    fig_radar = go.Figure(data=radar_data)
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Performance Metrics Radar Chart"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("F1 Score Comparison & Confusion Matrix")
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.markdown("**F1 Score Comparison**")
        model_names = list(results.keys())
        f1_scores = [results[m]["F1 Score"] for m in model_names]
        fig_f1 = px.bar(x=model_names, y=f1_scores,
                        labels={"x": "Model", "y": "F1 Score"},
                        title="F1 Score Comparison",
                        range_y=[0, 1],
                        color_discrete_sequence=[BAR_COLOR])
        fig_f1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_eval2:
        st.markdown("**Confusion Matrix (Simulated)**")
        np.random.seed(42)
        y_test_sim = np.random.randint(0, 2, size=100)
        y_pred_sim = np.random.randint(0, 2, size=100)
        cm = confusion_matrix(y_test_sim, y_pred_sim)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels={"x": "Predicted", "y": "Actual"},
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("ROC & Precision-Recall Curves (Simulated)")
    col_eval3, col_eval4 = st.columns(2)

    with col_eval3:
        st.write("**ROC Curve**")
        np.random.seed(42)
        y_test_sim = np.random.randint(0, 2, size=100)
        y_prob_sim = np.random.rand(100)
        fpr, tpr, _ = roc_curve(y_test_sim, y_prob_sim)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"ROC (AUC={roc_auc:.2f})",
                                     line=dict(width=LINE_WIDTH, color=LINE_COLOR)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                     name="Chance", line=dict(dash='dash', color='navy')))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0,1]),
            yaxis=dict(range=[0,1.05])
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_eval4:
        st.write("**Precision-Recall Curve**")
        np.random.seed(42)
        y_test_sim = np.random.randint(0, 2, size=100)
        y_prob_sim = np.random.rand(100)
        precision, recall, _ = precision_recall_curve(y_test_sim, y_prob_sim)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                    name="Precision-Recall",
                                    line=dict(width=LINE_WIDTH, color=LINE_COLOR)))
        fig_pr.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0,1]),
            yaxis=dict(range=[0,1])
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.success("Model evaluation dashboard loaded successfully!")
