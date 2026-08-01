import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encode inputs
sex = 1 if sex == "Male" else 0

if embarked == "C":
    embarked = 0
elif embarked == "Q":
    embarked = 1
else:
    embarked = 2

if st.button("Predict"):
    data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Not Survived")
