import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import json
import joblib

class EDVisitForecaster:
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def prepare_data(self, data_path):
        print("CSV data loading....")
        df = pd.read_csv(data_path)
        print(df.head())
        # Select only numeric columns for scaling
        numeric_df = df.select_dtypes(include=[np.number])
        print(numeric_df.head())
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric columns found in the input data for scaling.")
        scaled_data = self.scaler.fit_transform(numeric_df.values)
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i - self.window_size:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X)[..., np.newaxis], np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        self.model = model
        return model

    def train(self, X, y, epochs=20, batch_size=32):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

    def save_artifacts(self, model_path='ed_model.h5', scaler_path='scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_artifacts(self, model_path='ed_model.h5', scaler_path='scaler.pkl'):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler = joblib.load(scaler_path)

    def predict_next_hour(self, last_window):
        scaled_window = self.scaler.transform(np.array(last_window).reshape(-1, 1))
        input_data = scaled_window.reshape(1, self.window_size, 1)
        prediction = self.model.predict(input_data)
        return float(self.scaler.inverse_transform(prediction)[0][0])