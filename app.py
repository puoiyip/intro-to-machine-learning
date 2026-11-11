import streamlit as st
import joblib
import pandas as pd 

#load model 
model = joblib.load("heart_disease_decision_tree.joblib")

#set up page configuration (streamlit)
st.set_page_config(page_title="Heart Disease Predictor")
st.title("Heart Disease Predictor (made with Girls Who Code)")
st.subheader("Notes About the Project")
st.text("I created a decision tree that is intends on predicting the likelihood of a patient having heart disease based on a variety of health and demographic factors using data collected from 2020. This current project only shows the accuracy of my model (which was the extent of the project guided by Girls Who Code), but I want to use my model to predict the outome based on user inputs (this is a feature I'm currently working on).")

#original data without encoding
original_data = pd.read_csv("heartDisease_2020_sampling.csv")
st.subheader("Original Data")
st.text("This is the original data, collected from a Kaggle dataset that used CDC survey data.")
st.dataframe(original_data)

#encoded dataframe
df = joblib.load("heart_disease_dataframe.joblib") #loads encoded data
st.subheader("Encoded Data")
st.text("This text has been label encoded and one-label encoded. HeartDisease, GenHealth, Smoking, AlcoholDrinking, AgeCategory, and PhysicalActivity have all been label encoded. Sex and Race have been one-label encoded. Very few data came from individuals with heart disease, so the model was adjusted to give more priority to the minority class.")
st.dataframe(df)

#graph feature importance
feature_importance = joblib.load("feature_importance.joblib") #loads feature importance dataframe
st.subheader("Feature Importance")
st.text("The bar chart below shows the distribution of feature importance, the impact of each variable on the model's predictions. ")
st.bar_chart(feature_importance)

#accuracy of the models
test_acc = joblib.load("test_acc.joblib")
train_acc = joblib.load("train_acc.joblib")
st.subheader("Model Accuracy")
st.text("The accuracy with the testing data set of the Decision Tree is: " + str(test_acc))
st.text("The accuracy with the training data set of the Decision Tree is: " + str(train_acc))

#USER INPUT STUFF FOR NEW FEATURE

# #user input (risk factors for heart disease)
# bmi_input = st.text_input("What is your BMI?")
# smoking = st.selectbox("Do you smoke?", ["yes", "no"])
# alcohol_drinking = st.selectbox("Do you drink alcohol?", ["yes", "no"])
# sex = st.selectbox("What is your sex?", ["female", "male"])
# age_category = st.selectbox("Select your age category.", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
# race = st.multiselect("What is your race? Select all that apply.", ["American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "White", "Other"])
# physical_activity = st.selectbox("Are you physically active?", ["yes", "no"])
# gen_health = st.number_input("How good is your health, from 1 being poor health to 5 being excellent health?", 1, 5)

# #button
# if st.button("Analyze Data"):
#     bmi = float(bmi_input) if isinstance(bmi_input, str) else st.warning("Please enter a numeric value.")
#     user_data = {
#         "BMI": [bmi],
#         "Smoking": [smoking],
#         "AlcoholDrinking": [alcohol_drinking],
#         "Sex": [sex],
#         "AgeCategory": [age_category],
#         "Race": [race],
#         "PhysicalActivity": [physical_activity],
#         "GenHealth": [gen_health]
#     }
#     df = pd.DataFrame(user_data)
#     prediction = model.predict(user_data)
#     label = "likely to develop heart disease" if prediction else "not likely to develop heart disease"
#     st.success(f"Prediction: {label}")