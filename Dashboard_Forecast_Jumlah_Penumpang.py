import numpy as np
import matplotlib.pyplot as plt
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
st.title('Passenger Data Visualization')

# Display Data
st.write(data)

# Plotly Express Plot
fig = px.line(data, x='Tanggal', y='Jumlah_Penumpang', title='Jumlah Penumpang Over Time')

st.plotly_chart(fig)
