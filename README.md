# Forecast.fm — ARIMA-LSTM Hybrid Forecasting of Spotify Streams

This repository contains a music stream forecasting project that leverages a hybrid ARIMA-LSTM model to predict daily Spotify stream counts. It includes two core components:

1. A Jupyter notebook for detailed, song-specific modeling and evaluation.
2. A Streamlit web application that allows users to select any song from a predefined dataset and view forecasting results interactively.

---

## Project Overview

This project demonstrates a hybrid modeling approach to time series forecasting by combining:

- **ARIMA (AutoRegressive Integrated Moving Average)**: A classical statistical model used to capture linear components such as trend and seasonality in time series data.
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) capable of learning complex, nonlinear temporal dependencies. LSTMs are particularly effective for sequence modeling where long-term memory is essential.

By integrating these models, the system captures both the linear and nonlinear dynamics present in music streaming data, resulting in more accurate and robust predictions.

---

## Jupyter Notebook (`spotify_arima_lstm.ipynb`)

The notebook provides a structured pipeline for:

- Selecting a specific song (e.g., "Shape of You")
- Resampling and transforming the data for stationarity and variance stability
- Fitting an ARIMA model to model linear components
- Training an LSTM model on ARIMA residuals to capture nonlinearities
- Reconstructing the hybrid forecast
- Evaluating performance using RMSE, MAPE, and NRMSE

This notebook is intended for experimentation, diagnostics, and single-song analysis.

---

## Streamlit App (`app_streamlit.py`)

The Streamlit application offers an interactive interface that enables users to:

- Choose any song from the dataset
- Visualize historical daily stream counts
- View ARIMA-LSTM hybrid predictions
- Review evaluation metrics for forecasting accuracy

The app processes the selected song dynamically and applies the complete modeling pipeline in real-time. It is suitable for demos, internal tools, or educational dashboards.

---

## Model Summary

### ARIMA (AutoRegressive Integrated Moving Average)

ARIMA is a well-established statistical technique used for time series forecasting. It combines:

- **AutoRegressive (AR)**: Regression on lagged values
- **Integrated (I)**: Differencing to make the series stationary
- **Moving Average (MA)**: Modeling residuals from previous errors

ARIMA is most effective for time series that exhibit linear trends and seasonal patterns.

### LSTM (Long Short-Term Memory Networks)

LSTM is a deep learning model designed for sequential data. It addresses the limitations of traditional RNNs by using gated memory cells, enabling the model to retain long-term dependencies and handle vanishing gradients. LSTM is especially useful for modeling nonlinear temporal patterns that are difficult to capture with statistical models.

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/TheSkullKandy/Forecast.fm-ARIMA-LSTM-Hybrid-Forecasting.git
cd Forecast.fm-ARIMA-LSTM-Hybrid-Forecasting
```
---

### Install dependencies

```bash
pip install -r requirements.txt
```
---

### Run the Streamlit application

```bash
streamlit run app_streamlit.py
```
