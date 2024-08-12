#load model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
# from fbprophet import Prophet
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def model_arima(df):
    # Fit ARIMA model
    model_arima = ARIMA(df.iloc[:,-1], order=(5, 1, 0))
    model_arima_fit = model_arima.fit()

    return model_arima_fit

def model_es(df):
    # Fit Exponential Smoothing model
    model_holt = ExponentialSmoothing(df['Jumlah_Penumpang'],
                                  trend='add',  # options: 'add', 'mul', None
                                  seasonal='add',  # options: 'add', 'mul', None
                                  seasonal_periods=7,
                                  damped_trend=True,  # Damped trend
                                  initialization_method="estimated")

    model_holt_fit = model_holt.fit()

    return model_holt_fit

# def model_fbprophet(df):
#     # Fit FB Propet model
#     df_prophet = df.reset_index()
#     df_prophet.columns = ['ds', 'y']
#     # Fit Prophet model
#     model_prophet = Prophet()
#     model_prophet.fit(df_prophet)
#
#     return model_prophet


def model_LSTM(df):
    # Fit FB Propet model
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Prepare data for LSTM
    X, y = [], []
    for i in range(1, len(scaled_data)):
        X.append(scaled_data[i - 1:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model_lstm = tf.keras.models.Sequential()
    model_lstm.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model_lstm.add(tf.keras.layers.LSTM(units=50))
    model_lstm.add(tf.keras.layers.Dense(1))

    # Compile model
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model_lstm.fit(X[:-10], y[:-10], epochs=20, batch_size=1, verbose=2)

    # Prepare test data
    # test_input = scaled_data[-1, 0].reshape(-1, 1)
    # test_input = np.reshape(test_input, (test_input.shape[0], test_input.shape[1], 1))

    pred = model_lstm.predict(X[-10:])
    predicted_original = scaler.inverse_transform(pred)
    print(predicted_original)
    return predicted_original