import streamlit as st
import numpy as np
import joblib
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: 'diabetes_model.joblib' or 'scaler.joblib' not found. Please ensure they are in the same directory as this script.")

#page configuration
st.set_page_config(
    page_title="DiaSense - Smart Risk Analyzer",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

#Custom CSS Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 38px;
            color: #D9A6AD;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .title {
            text-align: center;
            font-size: 38px;
            color: #D8C7F0;
            margin-bottom: 30px;
        }

        .stButton>button {
            width: 100%;
            background-color: #F6C1C7;
            color: #4B4B4B;
            font-size: 18px;
            padding: 12px;
            border-radius: 18px;
        }

        .stButton>button:hover {
            background-color: #D9A6AD;
            color: white
        }

        .card {
            background-color: #FFF8E7;
            box-shadow: 1px 1px 10px #D9DDE2;
            padding: 20px;
            border-radius: 15px;
        }

        .footer{
            text-align : center;
            color: #4B4B4B;
            font-size: 14px;
            margin-top: 30px;

        }
    </style>
""",unsafe_allow_html=True)

#App header
st.markdown('<div class="title"> DiaSense</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle"> Smart Diabetes Risk Analyzer</div>',unsafe_allow_html=True)


#input section
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    preg=st.number_input("Pregnancies", 0,20, step=1)
    bp=st.number_input("Blood Pressure", 0,200)
    insulin=st.number_input("Insulin", 0,900)
    dpf=st.number_input("Diabetes Pedigree Function", 0.0,3.0)
with col2:
    glucose=st.number_input("Glucose", 0,200)
    skin=st.number_input("Skin Thickness", 0,100)
    bmi=st.number_input("BMI", 0.0,70.0)
    age=st.number_input("Age", 0,120)
st.markdown('</div>', unsafe_allow_html=True)

#prediction button

st.write("")
predict_btn= st.button("Predict Diabetes Risk")
if predict_btn:
    input_data= np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data= scaler.transform(input_data)
    pred= model.predict(input_data)[0]
    prob= model.predict_proba(input_data)[0][1]
    if pred==1:
        st.error(f"⚠️ HIGH RISK: The patient is likely diabetic\n\n *Probability:* {prob:.2f}")
    else:
        st.success(f" ✔️Low Risk: The patient is not diabetic \n\n *Probability:*{prob:.2f}")

#Footer with copyright
st.write("---")
st.markdown(
    "<p class='footer'? ©️ DiaSense- All Rights Reserved. </p>",unsafe_allow_html=True
)
