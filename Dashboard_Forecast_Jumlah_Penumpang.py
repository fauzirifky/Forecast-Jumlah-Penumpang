import numpy as np
import matplotlib.pyplot as plt
from settings import *
from model import *
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
from sklearn.linear_model import LinearRegression
import pymysql
import plotly.express as px
import os
from dotenv import load_dotenv
load_dotenv()



# Connect to the database
# Accessing secrets
try:
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')
except:
    db_host = st.secrets["database"]["DB_HOST"]
    db_user = st.secrets["database"]["DB_USER"]
    db_password = st.secrets["database"]["DB_PASSWORD"]
    db_name = st.secrets["database"]["DB_NAME"]

connection = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    db=db_name
)


# Query to fetch data
query = "SELECT * FROM passenger_data"

# Load data into a pandas DataFrame
data = pd.read_sql(query, connection)
print(data)
# Close the connection
connection.close()


# Streamlit App
custom_css = css()
st.markdown(custom_css, unsafe_allow_html=True)

st.title('Passenger Data Visualization')

# Display Data
st.write(data)

#Running model
ma = model_arima(data)
mes = model_es(data)
# mfp = model_fbprophet(data)

# Plotly Express Plot

fig = px.line(data.iloc[:,:-10], x='Tanggal', y='Jumlah_Penumpang', title='Jumlah Penumpang Over Time')

st.plotly_chart(fig)
