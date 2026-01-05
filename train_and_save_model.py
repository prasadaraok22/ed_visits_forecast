from ed_forecaster import EDVisitForecaster

forecaster = EDVisitForecaster(window_size=24)
X, y = forecaster.prepare_data('ed_patient_visits_data.csv')
print("ED Visits data loading complete.")
forecaster.build_model((X.shape[1], 1))
print("Model built.")
forecaster.train(X, y)
print("Model trained.")
forecaster.save_artifacts('ed_model.h5', 'scaler.pkl')
print("Model saved.")