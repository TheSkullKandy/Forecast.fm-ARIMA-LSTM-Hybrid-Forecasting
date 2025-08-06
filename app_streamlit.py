
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K

st.title("Forecast.fm")
st.subheader("Music Streams ARIMA-LSTM Hybrid Forecast")

# Loading dataset
df = pd.read_csv("data/data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# User chooses a song
song_options = df['Track Name'].unique()
selected_song = st.selectbox("Select a song", song_options)

# Filter data
song_df = df[df['Track Name'] == selected_song]
song_df = song_df.groupby('Date')['Streams'].sum().reset_index()
song_df = song_df.set_index('Date').asfreq('D').fillna(method='ffill')
song_df['LogStreams'] = np.log1p(song_df['Streams'])

# Show raw stream plot
st.subheader(f"📈 Daily Streams for '{selected_song}'")
st.line_chart(song_df['Streams'])

# --- ARIMA ---
arima_model = ARIMA(song_df['LogStreams'], order=(5,1,0))
arima_result = arima_model.fit()
arima_pred = arima_result.predict(start=1, end=len(song_df), typ='levels')
residuals = song_df['LogStreams'][1:] - arima_pred
residuals = residuals.dropna()

# --- LSTM ---
residuals = residuals.values.reshape(-1, 1)
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals)

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_len = 10
X, y = create_sequences(residuals_scaled, seq_len)
X = X.reshape((X.shape[0], X.shape[1], 1))

K.clear_session()
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_len, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=0)

predicted_residuals_scaled = model.predict(X)
predicted_residuals = scaler.inverse_transform(predicted_residuals_scaled)

# Combine predictions
min_len = min(len(arima_pred[seq_len+1:]), len(predicted_residuals))
aligned_arima = arima_pred[seq_len+1:][:min_len]
aligned_residuals = predicted_residuals[:min_len]
final_log_pred = aligned_arima + aligned_residuals.flatten()
actual_log = song_df['LogStreams'].values[seq_len+1:][:min_len]
final_pred = np.expm1(final_log_pred)
actual = np.expm1(actual_log)
aligned_dates = song_df.index[seq_len+1:][:min_len]

# --- Evaluation ---
mask = ~np.isnan(actual) & ~np.isnan(final_pred)
actual_clean = actual[mask]
final_pred_clean = final_pred[mask]
mape = np.mean(np.abs((actual_clean - final_pred_clean) / actual_clean)) * 100
rmse = np.sqrt(mean_squared_error(actual_clean, final_pred_clean))
nrmse = rmse / (actual_clean.max() - actual_clean.min())

# --- Output ---
st.subheader("Forecast Evaluation")
st.markdown(f"**MAPE:** {mape:.2f}%")
st.markdown(f"**RMSE:** {rmse:,.0f}")
st.markdown(f"**NRMSE:** {nrmse:.4f}")

st.subheader("Forecast vs Actual")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(aligned_dates, actual_clean, label="Actual Streams")
ax.plot(aligned_dates, final_pred_clean, label="Hybrid Prediction")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Streams")
ax.set_title(f"Hybrid Forecast for '{selected_song}'")
st.pyplot(fig)
