# Streamlit
import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.joblib')

#page configuration
st.set_page_config(page_title="Package Prediction", page_icon=":smiley:", layout="centered")

#title
st.title("Package Prediction")
st.write("This is a simple web application that predicts the package based on its cgpa.")

# input CGPA
cgpa = st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict"):
    try:
        # Prepare the input data for prediction
        input_data = np.array([[cgpa]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display the result
        st.success(f"The predicted package is: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")