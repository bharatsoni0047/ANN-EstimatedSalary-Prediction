import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,  StandardScaler, OneHotEncoder
import pickle
import warnings

# unnecessary warnings ko band kar diya (scikit-learn / tensorflow)
warnings.filterwarnings("ignore")

# loading the trainded model
model = tf.keras.models.load_model("Regression_model.h5")
with open("onehot_encoder_geo.pkl", "rb") as f:
    ohe = pickle.load(f)  # geography ka one-hot encoder
with open("label_encoder_gender.pkl", "rb") as f:
    le = pickle.load(f)   # gender ka label encoder
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)  # numerical values ka scaler


# Streamlit ka UI (user input lene ke liye)
st.title("💳Estimated Salary Prediction App") 
st.write("Niche details bharo aur turant churn probability dekh lo:") 

# User se input lena (alag-alag features)
geography = st.selectbox("🌍 Geography", ohe.categories_[0], key="geo_select")
gender = st.selectbox("👤 Gender", le.classes_, key="gender_select")
age = st.slider("🎂 Age", 18, 90, 30, key="age_slider")
balance = st.number_input("💰 Balance", 0.0, 1e7, 0.0, key="balance_input")
credit_score = st.number_input("💳 Credit Score", 0, 1000, 600, key="credit_input")
exited = st.selectbox("Exited", [0,1]) 
tenure = st.slider("📅 Tenure (Years with Bank)", 0, 10,1, key="tenure_slider")
num_products = st.slider("🛒 Number of Products", 1, 4, 1, key="products_slider")
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1], key="card_select")
is_active_member = st.selectbox("✅ Is Active Member", [0, 1], key="active_select")


# Input ko model ke liye taiyar karna
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [le.transform([gender])[0]],  # gender ko number me badalna
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited" : [exited]
})

# Geography ko one-hot encoding me badalna
geo_ohe = ohe.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_ohe, columns=ohe.get_feature_names_out(["Geography"]))
input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

# Pura input scale karna (same tarike se jaise training time pe kiya tha)
input_scaled = scaler.transform(input_df)

# Prediction nikalna
pred_salary = model.predict(input_scaled)[0][0]

# Result dikhana
st.write(f"Predicted Estimated Salary: {pred_salary:.2f}")