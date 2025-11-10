import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import gdown

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# --- تحميل الموديل تلقائي باستخدام gdown ---
url = "https://drive.google.com/uc?id=1iheHK6YKJvcXgKlipLnyQbU4Mcex1pdI"
gdown.download(url, "model.pkl", quiet=False)

model = joblib.load("model.pkl")

# --- واجهة رفع الملفات ---
st.sidebar.header("Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # --- تنبؤ الـChurn ---
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:,1]
    data["Churn_Predicted"] = predictions
    data["Churn_Probability"] = probabilities

    st.header("Customer Churn Dashboard")
    
    # --- KPIs ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Customers", len(data))
    col2.metric("Predicted Churn Rate", f"{predictions.mean()*100:.2f}%")
    col3.metric("Average Churn Probability", f"{probabilities.mean()*100:.2f}%")

    # --- مثال رسم بياني ---
    fig = px.histogram(data, x="Churn_Probability", nbins=20, title="Churn Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # --- Download Button ---
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv_data = convert_df(data)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name='churn_results.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to start the analysis.")
