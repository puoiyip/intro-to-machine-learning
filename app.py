import streamlit as st
import joblib

#load model 
model = joblib.load("heart_disease_decision_tree.joblib")

#set up page configuration (streamlit)
st.set_page_config(page_title="Heart Disease Predictor")
st.title("Heart Disease Predictor (made with Girls Who Code)")

#user input (risk factors for heart disease)
bmi = st.text_input("What is your BMI?")
smoking = st.selectbox("Do you smoke?", ["yes", "no"])
alcohol_drinking = st.selectbox("Do you drink alcohol?", ["yes", "no"])
sex = st.selectbox("What is your sex?", ["female", "male"])
physical_activity = st.selectbox("Are you physically active?", ["yes", "no"])
gen_health = st.number_input("How good is your health, from 1 being poor health to 5 being excellent health?", 1, 5)
race = st.multiselect("What is your race? Select all that apply.", ["American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "White", "Other"])

#button
#bmi, smoking, alcohol drinking, age category, physical activity, gen health, sex (one hot), race (one hot)
# if st.button("Analyze Data"):
