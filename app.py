import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes ML Project", layout="wide")

try:
    with open('all_models.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        rf_model = saved_data['rf_model']
        rf_acc = saved_data['rf_acc']
        kmeans = saved_data['kmeans_model']
        scaler = saved_data['scaler']
        k_score = saved_data['k_score']
        diabetic_cluster_id = saved_data['diabetic_cluster_id']
except FileNotFoundError:
    st.error("Please run 'save_all_models.py' first!")
    st.stop()

st.title("Diabetes Prediction: Supervised vs Unsupervised")
st.markdown("Comparing **Random Forest** (Label-based) and **K-Means** (Pattern-based) predictions.")

st.sidebar.header("Patient Input")
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

st.subheader("Patient Data")
st.dataframe(input_df)

if st.button("Predict using Both Models"):
    col1, col2 = st.columns(2)
    
    rf_pred = rf_model.predict(input_df)[0]
    rf_prob = rf_model.predict_proba(input_df)[0][1]
    
    with col1:
        st.header("1. Supervised Model")
        st.markdown("**Algorithm:** Random Forest")
        if rf_pred == 1:
            st.error(f"Prediction: DIABETIC")
            st.write(f"Confidence: {rf_prob:.2%}")
        else:
            st.success(f"Prediction: HEALTHY")
            st.write(f"Confidence: {(1-rf_prob):.2%}")
            
    input_scaled = scaler.transform(input_df)
    cluster_pred = kmeans.predict(input_scaled)[0]
    
    with col2:
        st.header("2. Unsupervised Model")
        st.markdown("**Algorithm:** K-Means Clustering")
        st.write(f"Assigned to: Cluster {cluster_pred}")
        
        if cluster_pred == diabetic_cluster_id:
            st.warning("Interpretation: This cluster contains mostly Diabetic patients.")
        else:
            st.info("Interpretation: This cluster contains mostly Healthy patients.")

    st.markdown("---")
    
    st.subheader("Model Performance Metrics")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.info(f"**Random Forest Accuracy:** {rf_acc:.2%}")
        st.caption("Tested on 20% of data. Measures how often the model is correct.")
        
    with m_col2:
        st.info(f"**K-Means Silhouette Score:** {k_score:.2f}")
        st.caption("Measures how distinct the clusters are (Range: -1 to 1).")

st.markdown("---")
st.subheader("Project Visualizations")
tabs = st.tabs(["Correlations", "Glucose Impact", "Confusion Matrix", "Cluster Analysis"])

with tabs[0]:
    st.image("assets/images/viz1_correlation.png", caption="Feature Correlation")
with tabs[1]:
    st.image("assets/images/viz3_glucose_box.png", caption="Glucose vs Outcome")
with tabs[2]:
    st.image("assets/images/model_supervised_confusion_matrix.png", caption="RF Confusion Matrix")
with tabs[3]:
    st.image("assets/images/model_unsupervised_comparison.png", caption="K-Means Clusters vs Actual")