import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
@st.cache_resource
def load_model_scaler():
    with open('model.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    return classifier, sc

classifier, sc = load_model_scaler()

st.title("Social Network Ads Prediction")
st.write("Predict whether a user will purchase a product based on Age and Estimated Salary.")

age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
salary = st.number_input("Estimated Salary", min_value=10000, max_value=200000, value=50000, step=1000)

if st.button("Predict"):
    # Apply scaling
    input_data = sc.transform([[age, salary]])
    
    # Make prediction
    prediction = classifier.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Result: **Purchased!** 🛍️")
    else:
        st.error("Result: **Not Purchased.** ❌")
