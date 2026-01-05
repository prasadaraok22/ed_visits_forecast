
from ed_forecaster import EDVisitForecaster

# Initialize and load model artifacts
forecaster = EDVisitForecaster(window_size=24)
forecaster.load_artifacts('ed_model.h5', 'scaler.pkl')

# Example test input (replace with actual recent 24-hour visit counts)
test_window = [10, 12, 15, 14, 13, 16, 18, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

# Make prediction
scaled_pred = forecaster.predict_next_hour(test_window)
print("Scaled predicted next hour visits:", scaled_pred)
# Inverse transform to get original scale
pred = forecaster.scaler.inverse_transform([scaled_pred])[0]
print("Predicted next hour visits:", pred)