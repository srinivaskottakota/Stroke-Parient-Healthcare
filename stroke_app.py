import numpy as np
import pandas as pd
import pickle
import streamlit as st
import base64
import os


st.set_page_config(page_title="Stroke Prediction Using Deep Learning", layout="wide")

# Ridge Regression class
class RidgeRegression:
    def __init__(self, lr=0.01, n_iters=1000, alpha=0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha  # Regularization strength
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha / num_samples) * self.weights
            db = (1 / num_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the correct directory.")
        st.stop()
    except pickle.UnpicklingError:
        st.error("Failed to load the model. The file might be corrupted.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()



model = load_model()


st.markdown("<h1 style='font-family:Playfair Display; font-weight: normal; color: #7f8c8d; text-align: center;'>Stroke Prediction Web Application</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-family:Playfair Display; font-weight: normal; color: white; text-align: center; font-size: 15px; margin-top: -10px;'>Leverage Deep Learning to Predict Stroke Risk Based on Key Health Factors</h3>", unsafe_allow_html=True)


# Define the layout proportions for image and input form (2 columns)
col1, col2 = st.columns([2.5, 2.5])  # Left column for image (3 parts), right column for input features (1 part)

# Image column (left side)
with col1:
   
    img_path = r"C:\Users\arath\Stroke\stroke5.jpg"
    if os.path.exists(img_path):
        with open(img_path, "rb") as file:
            img_data = file.read()
        encoded_img = base64.b64encode(img_data).decode("utf-8")
        st.markdown(
            """
            <style>
            .custom-img {
                width: 100%; /* Full width of the column */
                height: 100vh; /* Full viewport height */
                object-fit: cover; /* Maintain aspect ratio, cropping excess */
                border-radius: 10px; /* Optional: rounded corners */
                padding-top: 10px;
                padding-left: 1px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<img src="data:image/jpeg;base64,{encoded_img}" class="custom-img">',
            unsafe_allow_html=True,
        )
    else:
        st.error("Image file not found. Please check the file path.")

# Input features column (right side)
with col2:
    
    st.sidebar.header("About")
    st.sidebar.markdown(""" 
    This Streamlit app is designed to predict the likelihood of a stroke based on several health factors using a deep learning model. 
    The app takes input features such as age, gender, hypertension, heart disease, marital status, work type, and smoking status, 
    and provides a prediction on the probability of a stroke. The deep learning model has been trained on a dataset of health records 
    and aims to assist healthcare professionals in identifying individuals at risk of stroke. 

    The model is based on deep learning algorithms, which analyze patterns in the data to make predictions.
    """, unsafe_allow_html=True)

    # Collect user inputs for the features
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, value=30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, value=100.0)
    bmi = st.number_input("Body Mass Index", min_value=10.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    # Encode categorical inputs manually
    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    work_type = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"].index(work_type)
    residence_type = 1 if residence_type == "Urban" else 0
    smoking_status = ["never smoked", "formerly smoked", "smokes"].index(smoking_status)

    # Input features (10 features + dummy feature)
    inputs = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
    inputs_with_dummy = np.hstack([inputs, np.zeros((inputs.shape[0], 1))])  # Add the dummy feature

    # Buttons for Predict and Reset
    col2_button1, col2_button2 = st.columns([1, 1])  # Create two columns for buttons
    with col2_button1:
        predict_button = st.button("Predict")

    with col2_button2:
        reset_button = st.button("Reset")


if predict_button:
    try:
        prediction = model.predict(inputs_with_dummy)
        prediction_class = "Stroke" if prediction[0] >= 0.5 else "No Stroke"
        
        # Define the message and color based on prediction
        if prediction_class == "Stroke":
            message = "Prediction: <br> A Stroke is likely. Please consult a healthcare professional for further evaluation."
            message_color = "white"  # Text color for stroke prediction
            font_size = "18px"  # Font size for stroke prediction
            background_color = "#272630"  # Light red background for Stroke prediction (danger)
        else:
            message = "Prediction: <br> No Stroke detected. Keep maintaining a healthy lifestyle!"
            message_color = "white"  # Text color for no stroke prediction
            font_size = "18px"  # Font size for no stroke prediction
            background_color = "#272630"  # Green background for No Stroke prediction (safe)

        # Display prediction message with background color, color, and font size styling
        with col1:
            st.markdown(f"""
                <div style='
                    font-family:Proxima Nova;
                    font-weight: normal;
                    color:{message_color};
                    font-size:{font_size};
                    text-align: center;
                    background-color:{background_color};
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                '>{message}</div>
            """, unsafe_allow_html=True)
           
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Reset functionality
if reset_button:
    st.session_state.clear()  # Clear session state (input values)
    st.cache_data.clear()  # Clear any cached data
    st.stop()  # Stop execution and reload the page


# Custom CSS for hover effect on buttons
st.markdown("""
    <style>
    button[title="Predict"], button[title="Reset"] {
        background-color: #f0f0f0; /* Light background color */
        border: 1px solid #d1d1d1; /* Light border */
        color: #000000; /* Text color */
    }

    button[title="Predict"]:hover, button[title="Reset"]:hover {
        background-color: #ffffff; /* White background on hover */
        border: 1px solid #d1d1d1; /* Keep border color */
    }

    /* Additional styling for other elements can be added here */
    </style>
""", unsafe_allow_html=True)
