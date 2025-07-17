import unittest
import pandas as pd
import json
import os
import tempfile
import shutil
from data_degradation_detector import univariate as uv
import data_degradation_detector.report as report


class TestInforms(unittest.TestCase):
    """Unit tests for the informs module and related functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample DataFrame for testing
        self.df = pd.read_csv("data/WineQT.csv")
        self.X = self.df.drop(columns=["quality", "Id"], axis=1)
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and its contents
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_initial_inform(self):
        """Test the create_initial_inform function."""
        # Call the function to create initial inform
        report.create_initial_inform(self.X, self.temp_dir)
        
        # Check if the JSON file was created
        json_file_path = os.path.join(self.temp_dir, "distribution_descriptors.json")
        self.assertTrue(os.path.exists(json_file_path), 
                       "distribution_descriptors.json file should be created")
        
        # Check if the PNG file was created
        png_file_path = os.path.join(self.temp_dir, "distribution_descriptors_all_columns.png")
        self.assertTrue(os.path.exists(png_file_path), 
                       "distribution_descriptors_all_columns.png file should be created")
        
        # Verify that the JSON file contains valid data
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        self.assertIsInstance(json_data, dict, "JSON data should be a dictionary")
        self.assertEqual(len(json_data), len(self.X.columns), 
                        "JSON should contain data for all columns")
        
        # Check that each column has the expected statistical descriptors
        for column_name in self.X.columns:
            self.assertIn(column_name, json_data, 
                         f"Column {column_name} should be in JSON data")
            
            column_data = json_data[column_name]
            expected_keys = ['mean', 'std', 'min_val', 'max_val', 'q1', 'q2', 'q3']
            for key in expected_keys:
                self.assertIn(key, column_data, 
                             f"Key {key} should be in column data for {column_name}")
                self.assertIsInstance(column_data[key], (int, float), 
                                    f"Value for {key} should be numeric")

if __name__ == '__main__':
    # Run the unit tests
    unittest.main(verbosity=2)
