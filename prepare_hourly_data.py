import pandas as pd

def process_hourly_data(file_path):
    # 1. Load the CSV file into a Pandas DataFrame
    # Use parse_dates to automatically convert the date column upon loading (optional but efficient)
    df = pd.read_csv(file_path, parse_dates=['admission_date_time'])

    # If you didn't use parse_dates above, convert the column to datetime objects
    # df['datetime_column'] = pd.to_datetime(df['datetime_column'])

    # 2. Group the data by the category column and resample hourly
    # Use pd.Grouper(key='datetime_column', freq='H') for time-based grouping
    # Then apply an aggregation function, such as .mean()
    grouped_hourly_data = df.groupby([
        'facility_id',
        pd.Grouper(key='admission_date_time', freq='h')
    ])['patient_id'].count() # Calculate the mean of 'value_column' for each group

    # The result will be a Series with a MultiIndex.
    # You might want to reset the index to make it a conventional DataFrame
    result_df = grouped_hourly_data.reset_index()

    return result_df

# Example usage
file_path = 'ed_patient_visits_data.csv'
hourly_summary = process_hourly_data(file_path)
print(hourly_summary)
