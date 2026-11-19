
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

st.set_page_config(layout='wide')

# This code resolves to the directory containing App.py

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "XGBOOST_Churn_Model.pkl"
churn_image_path = BASE_DIR / 'customer_churn_image.jpeg'
customer_image_path = BASE_DIR / 'customer.jpg'

model = joblib.load(model_path)

st.title('Welcome to Churn Prediction App')

# Welcome image
picture = Image.open(churn_image_path).resize((1000,300))
st.image(picture,width= 'stretch')

# Sidebar image
customer = Image.open(customer_image_path).resize((300,200))
st.sidebar.image(customer,width= 'stretch')

# APP Welcome question.
st.divider()
st.markdown('### Please enter the customer details on the left bar to predict whether a customer will churn or not.')

st.divider()

# This code block creates two columns in the Streamlit layout.
col_1,col_2 = st.columns(2)

# This code block creates the sidebar for user input, allowing users to enter customer details.

st.sidebar.header('Customer Details')
gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
subscription_type = st.sidebar.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
contract_length = st.sidebar.selectbox('Contract Length', ['Monthly', 'Quarterly', 'Annual'])  

age = st.sidebar.number_input('Age', min_value=18, max_value=65, value=30)
tenure = st.sidebar.number_input('Tenure (in months)', min_value=1, max_value=60, value=12) 
usage_frequency = st.sidebar.number_input('Usage Frequency (in the last month)', min_value=1, max_value=30, value=5)
support_calls = st.sidebar.number_input('Support Calls(in the last month)', min_value=0, max_value=10, value=1)
payment_delay = st.sidebar.number_input('Payment Delay (days in the last month)', min_value=0, max_value=30, value=0)
total_spend = st.sidebar.number_input('Total Spend ', min_value=100, max_value=1000, value=300)
last_interaction = st.sidebar.number_input('Last Interaction (days ago)', min_value=1, max_value=30, value=15)


# This code block displays the feature importances chart in the second column of the layout.

with col_2:
        df = pd.read_csv(BASE_DIR / "impotances_dataframe.csv"))

        fig = px.bar(x = df.iloc[0], y = df.columns, title = 'Features Importances plot',
        width = 900, height = 500,
        labels = {'x':'Feature Importances','y':'Features'},
        color = df.columns,
        color_discrete_sequence = px.colors.qualitative.Vivid,
        hover_name = df.columns
      ).update_layout(showlegend=False)
        
        st.plotly_chart(fig,theme="streamlit", use_container_width=True)


# This code block handles the prediction logic when the user clicks the "Predict Churn" button in the first column of the layout.

with col_1:
    button = st.button('Predict Churn',type ='primary')
    if button:
      
        input_data = pd.DataFrame({
            'Gender': gender,
            'Subscription Type': subscription_type,
            'Contract Length': contract_length,
            'Age': age,
            'Tenure': tenure,
            'Usage Frequency': usage_frequency,
            'Support Calls': support_calls,
            'Payment Delay': payment_delay,
            'Total Spend': total_spend,
            'Last Interaction': last_interaction
        },index=[1])

        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data) 

        st.markdown(f'**The predicted churn value is: {int(prediction[0])}**')
        
        if prediction[0] == 1:
            p1 = 100*probability[0,1]
            st.error(f'**The customer is likely to churn.**')
            st.error(f'**The probability to churn is {p1:.2f} %.**')
        else:
            p0 = 100*probability[0,0]
            st.balloons() 
            st.success(f'**The customer is not likely to Churn.**')
            st.success(f'**The probabiity to Not-Churn is {p0:.2f} %.**')

            
    else:
        st.markdown('**Please fill in all the details and click on Predict Churn button.**')      
