import pandas as pd
from ed_forecaster import EDVisitForecaster


def get_facility_forecast(facility_id, csv_path='ed_patient_visits_data.csv'):
    # 1. Initialize Forecaster
    forecaster = EDVisitForecaster(window_size=24)
    try:
        forecaster.load_artifacts('ed_model.h5', 'scaler.pkl')
    except Exception as e:
        return f"Error loading model: Ensure you have trained the model first. {e}"

    # 2. Extract most recent history for the facility
    df = pd.read_csv(csv_path)
    df['admission_date_time'] = pd.to_datetime(df['admission_date_time'])

    # Filter and Aggregate
    facility_df = df[df['facility_id'].astype(str) == str(facility_id)]
    if facility_df.empty:
        return f"No data found for facility_id: {facility_id}"

    hourly_series = facility_df.set_index('admission_date_time').resample('h').size()

    # 3. Get the last 24 hours (the input window)
    if len(hourly_series) < 24:
        return f"Insufficient history for facility {facility_id}. Need 24 hours, have {len(hourly_series)}."

    last_window = hourly_series.tail(24).values

    # 4. Predict
    prediction = forecaster.predict_next_hour(last_window)
    return round(prediction, 2)


if __name__ == "__main__":
    # Example: Check prediction for PeaceHealth (ID: 3798668)
    target_id = "997424"
    forecast = get_facility_forecast(target_id)
    print(f"Facility {target_id} Forecast for next hour: {forecast} visits")