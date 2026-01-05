import pandas as pd


def convert_json_to_csv_specific_cols(json_file_path, csv_file_path, columns_to_keep):
    """
    Converts a JSON file to a CSV file, keeping only specific columns.

    Args:
        json_file_path (str): The path to the input JSON file.
        csv_file_path (str): The path for the output CSV file.
        columns_to_keep (list): A list of column names (strings) to retain.
    """
    try:
        # Load the JSON data into a DataFrame
        df = pd.read_json(json_file_path)

        # Select the specific columns
        df_selected = df[columns_to_keep]

        # Export the selected data to a CSV file
        df_selected.to_csv(csv_file_path, index=False)

        print(f"Successfully converted '{json_file_path}' to '{csv_file_path}' with columns: {columns_to_keep}")

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except KeyError as e:
        print(f"Error: One or more specified columns were not found in the JSON data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage:
input_json = 'ed_patient_visits_data.json'
output_csv = 'ed_patient_visits_data.csv'
desired_columns = ['facility_id', 'patient_id', 'admission_date_time']

convert_json_to_csv_specific_cols(input_json, output_csv, desired_columns)
