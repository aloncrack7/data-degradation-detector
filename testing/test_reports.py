import unittest
import pandas as pd
import json
import os
import tempfile
import shutil
from data_degradation_detector import univariate as uv
from data_degradation_detector import multivariate as mv
import data_degradation_detector.report as report


class TestReports(unittest.TestCase):
    """Unit tests for the reports module and related functionality."""
    
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
    
    def test_create_initial_report(self):
        """Test the create_initial_report function."""
        # Call the function to create initial report
        report.create_initial_report(self.X, self.temp_dir)
        
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
    
    def test_create_initial_report_with_clusters(self):
        """Test the create_initial_report function with clustering functionality."""
        # Load the basic1.csv data for clustering test
        df_basic = pd.read_csv("data/basic1.csv")
        X_basic = df_basic[["x", "y"]]
        
        # Call the function to create initial report with clustering
        report.create_initial_report(X_basic, self.temp_dir)
        
        # Check if the clustering JSON file was created
        cluster_json_path = os.path.join(self.temp_dir, "kmeans_clusters.json")
        self.assertTrue(os.path.exists(cluster_json_path), 
                       "kmeans_clusters.json file should be created")
        
        # Check if the clustering PNG file was created
        cluster_png_path = os.path.join(self.temp_dir, "kmeans_clusters_4.png")
        self.assertTrue(os.path.exists(cluster_png_path), 
                       "kmeans_clusters_4.png file should be created")
        
        # Verify that the clustering JSON file contains valid data
        with open(cluster_json_path, 'r') as f:
            clustering_data = json.load(f)
        
        self.assertIsInstance(clustering_data, dict, "Clustering data should be a dictionary")
        
        # Check if the number of clusters is 4
        num_clusters = clustering_data.get('num_clusters', 0)
        self.assertEqual(num_clusters, 4, "Number of clusters should be 4")
        
        # Check that the clustering data has all expected keys
        expected_keys = ['num_clusters', 'silhouette_score', 'centroids', 'radius', 'labels_percentages']
        for key in expected_keys:
            self.assertIn(key, clustering_data, 
                         f"Key {key} should be in clustering data")
        
        # Verify data types
        self.assertIsInstance(clustering_data['num_clusters'], int, 
                             "num_clusters should be an integer")
        self.assertIsInstance(clustering_data['silhouette_score'], (int, float), 
                             "silhouette_score should be numeric")
        self.assertIsInstance(clustering_data['centroids'], list, 
                             "centroids should be a list")
        self.assertIsInstance(clustering_data['radius'], list, 
                             "radius should be a list")
        self.assertIsInstance(clustering_data['labels_percentages'], list, 
                             "labels_percentages should be a list")
        
        # Test direct clustering vs JSON clustering consistency
        temp_cluster_path = os.path.join(self.temp_dir, "temp_cluster")
        clustering_direct = mv.get_cluster_defined_number(X_basic, 4, path=temp_cluster_path)
        # The temp_cluster_path file will be deleted in tearDown
        descriptors_json = mv.get_cluster_info_from_json(clustering_data)
        
        # Compare key attributes (allowing for small floating point differences)
        self.assertEqual(clustering_direct.num_clusters, descriptors_json.num_clusters,
                        "Number of clusters should match between direct and JSON methods")
        self.assertAlmostEqual(clustering_direct.silhouette_score, descriptors_json.silhouette_score, 
                              places=4, msg="Silhouette scores should be approximately equal")
        self.assertEqual(len(clustering_direct.centroids), len(descriptors_json.centroids),
                        "Number of centroids should match")
        self.assertEqual(len(clustering_direct.radius), len(descriptors_json.radius),
                        "Number of radius values should match")

        self.assertTrue(clustering_direct == descriptors_json, "Direct clustering and JSON clustering should be equal")

    def test_get_number_of_output_classes(self):
        """Test the get_number_of_output_classes function."""
        df = pd.read_csv("data/WineQT.csv")
        y = df["quality"]
        num_classes = report.get_number_of_output_classes(y)
        self.assertEqual(num_classes, 6, "Number of output classes should be 6")

        df = pd.read_csv("data/basic1.csv")
        y = df["color"]
        num_classes = report.get_number_of_output_classes(y)
        self.assertEqual(num_classes, 4, "Number of output classes should be 4")

        # Test with a Series that has more than 10 unique values
        y_large = pd.Series(range(15))
        num_classes = report.get_number_of_output_classes(y_large)
        self.assertIsNone(num_classes, "Number of output classes should be None for more than 10 unique values")

if __name__ == '__main__':
    # Run the unit tests
    unittest.main(verbosity=2)
