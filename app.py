import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes ML Project", layout="wide")

# --- Load All Models ---
try:
    with open('all_models.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        
        # Supervised Models
        rf_model = saved_data['rf_model']
        rf_acc = saved_data['rf_acc']
        lr_model = saved_data.get('lr_model') # Use .get() to avoid errors if key missing
        lr_acc = saved_data.get('lr_acc')
        svm_model = saved_data.get('svm_model')
        svm_acc = saved_data.get('svm_acc')
        
        # Unsupervised & Utils
        kmeans = saved_data['kmeans_model']
        scaler = saved_data['scaler']
        k_score = saved_data['k_score']
        diabetic_cluster_id = saved_data['diabetic_cluster_id']
except FileNotFoundError:
    st.error("Please run 'save_all_models.py' first!")
    st.stop()

st.title("🩺 Diabetes Prediction System")
st.markdown("Compare **Supervised Learning** (Classification) vs **Unsupervised Learning** (Clustering).")

# --- Sidebar Inputs ---
st.sidebar.header("1. Model Configuration")

# Dropdown to select the Supervised Model
model_choice = st.sidebar.selectbox(
    "Choose Supervised Model",
    ("Random Forest", "Logistic Regression", "Support Vector Machine (SVM)")
)

st.sidebar.header("2. Patient Data")
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 33.0)
    dpf = st.sidebar.slider('Diabetes Pedigree', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    return pd.DataFrame({
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
        'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }, index=[0])

input_df = user_input_features()

# --- Main Page Layout ---
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Patient Vitals")
    st.write(input_df.T) # Transposed for better view

# --- Logic to Select Model ---
if model_choice == "Random Forest":
    active_model = rf_model
    active_acc = rf_acc
    # Random Forest doesn't need scaling technically, but we pass raw data
    prediction_input = input_df 
elif model_choice == "Logistic Regression":
    active_model = lr_model
    active_acc = lr_acc
    # LR needs scaled data
    prediction_input = scaler.transform(input_df)
elif model_choice == "Support Vector Machine (SVM)":
    active_model = svm_model
    active_acc = svm_acc
    # SVM needs scaled data
    prediction_input = scaler.transform(input_df)

# --- Prediction Section ---
if st.button("Run Prediction Analysis"):
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    # 1. Supervised Prediction
    sup_pred = active_model.predict(prediction_input)[0]
    sup_prob = active_model.predict_proba(prediction_input)[0][1]
    
    with res_col1:
        st.header(f"1. {model_choice}")
        st.caption("Supervised Learning (Label-Based)")
        
        if sup_pred == 1:
            st.error("Prediction: DIABETIC")
            st.write(f"**Confidence:** {sup_prob:.2%}")
        else:
            st.success("Prediction: HEALTHY")
            st.write(f"**Confidence:** {(1-sup_prob):.2%}")
        
        st.info(f"Model Accuracy on Test Data: **{active_acc:.2%}**")

    # 2. Unsupervised Prediction (Always K-Means)
    # K-Means always needs scaled data
    input_scaled = scaler.transform(input_df)
    cluster_pred = kmeans.predict(input_scaled)[0]
    
    with res_col2:
        st.header("2. K-Means Clustering")
        st.caption("Unsupervised Learning (Pattern-Based)")
        st.metric("Assigned Cluster", f"Cluster {cluster_pred}")
        
        if cluster_pred == diabetic_cluster_id:
            st.warning("⚠️ This patient falls into the 'High Risk' cluster (typically Diabetic).")
        else:
            st.success("✅ This patient falls into the 'Low Risk' cluster (typically Healthy).")
            
        st.info(f"Silhouette Score (Cluster Quality): **{k_score:.2f}**")

# --- Visualizations ---
st.divider()
st.subheader("Project Visualizations")
tabs = st.tabs(["Correlations", "Glucose Impact", "Confusion Matrix (RF)", "Cluster Analysis"])

with tabs[0]:
    st.image("assets/images/viz1_correlation.png", caption="Feature Correlation Heatmap")
with tabs[1]:
    st.image("assets/images/viz3_glucose_box.png", caption="Glucose Levels: Diabetic vs Healthy")
with tabs[2]:
    st.image("assets/images/model_supervised_confusion_matrix.png", caption="Random Forest Confusion Matrix")
with tabs[3]:
    st.image("assets/images/model_unsupervised_comparison.png", caption="K-Means Clusters vs Actual Outcomes")