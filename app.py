
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Haraz-5 Well Log HC Zone App", layout="wide")

st.title("🛢️ Haraz-5 Well Log | HC Zone Prediction (Runtime Model)")

# Load training data from CSV
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("Haraz-5_clustered_with_HCZONE_and_Sw.csv")
    features = df[['GR', 'DT', 'RD', 'ZDEN']]
    target = df['HC_ZONE']

    # Impute and scale
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(features)
    X_scaled = scaler.fit_transform(X_imputed)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, target)

    return model, imputer, scaler

model, imputer, scaler = load_and_train_model()

tab1, tab2 = st.tabs(["📥 Predict from Uploaded CSV", "🎛 Real-Time Prediction"])

with tab1:
    st.header("📄 Upload Well Log CSV (GR, DT, RD, ZDEN)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if not all(col in df.columns for col in ['GR', 'DT', 'RD', 'ZDEN']):
            st.error("CSV must include GR, DT, RD, ZDEN columns.")
        else:
            X = df[['GR', 'DT', 'RD', 'ZDEN']]
            X_imp = imputer.transform(X)
            X_scaled = scaler.transform(X_imp)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            df['HC_ZONE_Pred'] = preds
            df['Probability'] = probs

            st.success("✅ Prediction completed.")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv, "HC_Predictions.csv", "text/csv")

            if 'DEPTH' in df.columns:
                st.subheader("📈 HC_ZONE vs Depth")
                fig, ax = plt.subplots(figsize=(5, 10))
                ax.plot(df['HC_ZONE_Pred'], df['DEPTH'], label="HC_ZONE_Pred", color="orange")
                ax.invert_yaxis()
                ax.set_xlabel("HC_ZONE_Pred")
                ax.set_ylabel("Depth (m)")
                st.pyplot(fig)

with tab2:
    st.header("🎛 Predict HC_ZONE from Manual Entry")
    col1, col2 = st.columns(2)
    with col1:
        gr = st.number_input("Gamma Ray (GR)", value=80.0)
        dt = st.number_input("DT (Sonic)", value=100.0)
    with col2:
        rd = st.number_input("Resistivity (RD)", value=30.0)
        zden = st.number_input("Bulk Density (ZDEN)", value=2.3)

    if st.button("🔍 Predict HC_ZONE"):
        input_df = pd.DataFrame([[gr, dt, rd, zden]], columns=['GR', 'DT', 'RD', 'ZDEN'])
        X_imp = imputer.transform(input_df)
        X_scaled = scaler.transform(X_imp)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        st.success(f"🔮 Prediction: {'Hydrocarbon Zone' if pred == 1 else 'Non-HC Zone'}")
        st.write(f"🧪 Confidence: {prob:.2%}")
