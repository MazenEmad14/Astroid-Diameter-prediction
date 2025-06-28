import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Inject custom CSS for a space-like theme
st.markdown("""
    <style>
        body {
            background-color: #0b0f2e;
            color: #e0e0e0;
        }
        .main {
            background-color: #0b0f2e;
        }
        h1, h2, h3 {
            color: #a3d9ff;
            text-align: center;
        }
        .planet {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #84a9ff, #001f4d);
            margin: 30px auto;
            position: relative;
            box-shadow: 0 0 30px #84a9ff;
        }
        .ring {
            position: absolute;
            top: 90px;
            left: -20px;
            width: 240px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(to right, #a3d9ff, #e0e0e0, #a3d9ff);
            transform: rotate(25deg);
            opacity: 0.6;
        }
        .stButton button {
            background-color: #2d73ff;
            color: white;
        }
        .stSuccess {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Show custom planet
st.markdown('<div class="planet"><div class="ring"></div></div>', unsafe_allow_html=True)

# Title
st.title("Asteroid Diameter Predictor")

st.markdown("""
This app predicts the diameter of an asteroid based on its orbital and physical parameters.
Enter the inputs below and click Predict.
""")

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    sc = pickle.load(f)

# Input UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("Categorical Features")
    neo = st.selectbox("Near-Earth Object (neo)", ['N', 'Y'])
    pha = st.selectbox("Potentially Hazardous Asteroid (pha)", ['N', 'Y'])
    asteroid_class = st.selectbox("Asteroid Class", list(le.classes_))

with col2:
    st.subheader("Numerical Features")
    H = st.number_input("Absolute Magnitude (H)", -10.0, 50.0, 5.0, step=0.1)
    albedo = st.number_input("Albedo", 0.0, 1.0, 0.1, step=0.01)
    epoch = st.number_input("Epoch (MJD)", 0.0, value=59000.0)
    e = st.number_input("Eccentricity (e)", 0.0, 1.0, 0.1, step=0.01)
    a = st.number_input("Semi-major Axis (a, AU)", 0.0, value=2.5)
    q = st.number_input("Perihelion Distance (q, AU)", 0.0, value=2.0)
    i = st.number_input("Inclination (i, degrees)", 0.0, value=10.0)
    om = st.number_input("Longitude of Ascending Node (om)", 0.0, 360.0, 0.0)
    w = st.number_input("Argument of Perihelion (w)", 0.0, 360.0, 0.0)
    ma = st.number_input("Mean Anomaly (ma)", 0.0, 360.0, 0.0)
    ad = st.number_input("Aphelion Distance (ad)", 0.0, value=3.0)
    rms = st.number_input("RMS Residual", 0.0, value=0.5, step=0.01)

if st.button("Predict Diameter"):
    try:
        input_dict = {
            'neo': [1 if neo == 'Y' else 0],
            'pha': [1 if pha == 'Y' else 0],
            'H': [H],
            'albedo': [albedo],
            'epoch': [epoch],
            'e': [e],
            'a': [a],
            'q': [q],
            'i': [i],
            'om': [om],
            'w': [w],
            'ma': [ma],
            'ad': [ad],
            'rms': [rms],
        }

        input_df = pd.DataFrame(input_dict)

        # One-hot for class
        class_encoded = {f'class_{cls}': 0 for cls in le.classes_}
        class_label = le.transform([asteroid_class])[0]
        class_encoded[f'class_{le.classes_[class_label]}'] = 1
        class_df = pd.DataFrame([class_encoded])

        input_df = pd.concat([input_df, class_df], axis=1)

        numeric_cols = ['H', 'albedo', 'epoch', 'e', 'a', 'q', 'i', 'om', 'w', 'ma', 'ad', 'rms']
        input_df[numeric_cols] = sc.transform(input_df[numeric_cols])

        expected_order = ['neo', 'pha', 'H', 'albedo', 'epoch', 'e', 'a', 'q', 'i', 'om', 'w', 'ma', 'ad', 'rms'] + \
                         [f'class_{cls}' for cls in le.classes_]
        input_df = input_df[expected_order]

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Diameter: {prediction:.2f} km")

    except Exception as e:
        st.error(f"Prediction error: {e}")
