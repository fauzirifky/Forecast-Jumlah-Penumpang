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
import plotly.graph_objects as go
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
st.write(data.iloc[:-10])

#Running model
ma = model_arima(data.iloc[:-10])
mes = model_es(data.iloc[:-10])
# mfp = model_fbprophet(data)
ma_predict = ma.forecast(steps=10)
mes_predict = mes.forecast(steps=10)
y = data['Jumlah_Penumpang'].values
y = y.reshape(-1,1)
mlstm = model_LSTM(y)
yl = mlstm.reshape(1,-1)
yl = yl[0]
# Plotly Express Plot

st.markdown('# Multiple Forecast Results')
fig = go.Figure()
# Add the original data plot (excluding the last 10 entries)
fig.add_trace(go.Scatter(x=data.iloc[-50:-10]['Tanggal'], y=data.iloc[-100:-10]['Jumlah_Penumpang'],
                         mode='lines', name='Actual Jumlah Penumpang'))
# Add the predicted data plot (for the last 10 entries)
fig.add_trace(go.Scatter(x=data.iloc[-10:]['Tanggal'], y=ma_predict,
                         mode='lines', name='Model ARIMA'))
fig.add_trace(go.Scatter(x=data.iloc[-10:]['Tanggal'], y=mes_predict,
                         mode='lines', name='Model Exponential Smoothing'))
fig.add_trace(go.Scatter(x=data.iloc[-10:]['Tanggal'], y=yl,
                         mode='lines', name='Model LSTM'))
fig.update_layout(
    title='Jumlah Penumpang Over Time',
    xaxis_title='Tanggal',
    yaxis_title='Jumlah Penumpang',
    legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Align the bottom of the legend to the plot
        y=-0.4,            # Position it below the x-axis
        xanchor="center",  # Center the legend horizontally
        x=0.5              # Center it within the plot area
    )
)
st.plotly_chart(fig)
