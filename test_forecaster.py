import unittest
import numpy as np
import os
import pandas as pd
from ed_forecaster import EDVisitForecaster

class TestEDVisitForecaster(unittest.TestCase):
    def setUp(self):
        self.window_size = 5
        self.forecaster = EDVisitForecaster(window_size=self.window_size)
        self.test_csv = 'test_data.csv'
        
        # Create dummy data: 20 hours of visits
        dates = pd.date_range('2026-01-01', periods=20, freq='h')
        data = {
            'facility_id': ['3798668'] * 20,
            'admission_date_time': dates
        }
        pd.DataFrame(data).to_csv(self.test_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_csv): os.remove(self.test_csv)
        if os.path.exists('temp_model.h5'): os.remove('temp_model.h5')
        if os.path.exists('temp_scaler.pkl'): os.remove('temp_scaler.pkl')

    def test_prediction_output_format(self):
        # 1. Prepare
        X, y = self.forecaster.prepare_data(self.test_csv)
        self.forecaster.build_model((X.shape[1], 1))
        self.forecaster.train(X, y, epochs=1)
        
        # 2. Predict
        mock_recent_counts = [1, 2, 1, 3, 2] # 5 values for window_size
        prediction = self.forecaster.predict_next_hour(mock_recent_counts)
        
        # 3. Validate
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0, "Predicted visits cannot be negative")

if __name__ == '__main__':
    unittest.main()
