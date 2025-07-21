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

        self.base_metrics = {
            "rmse": 0.6882765026767026,
            "r2": 0.17106058803680269,
            "mae": 0.5595104318406607
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and its contents
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_initial_report(self):
        """Test the create_initial_report function."""
        # Call the function to create initial report
        report.create_initial_report(self.X, self.base_metrics, self.temp_dir)

        # Check if the JSON file was created
        json_file_path = os.path.join(self.temp_dir, "distribution_descriptors.json")
        self.assertTrue(os.path.exists(json_file_path), 
                       "distribution_descriptors.json file should be created")
        
        # Check if the PNG file was created
        png_file_path = os.path.join(self.temp_dir, "distribution_descriptors_all_columns.png")
        self.assertTrue(os.path.exists(png_file_path), 
                       "distribution_descriptors_all_columns.png file should be created")
        
        base_metrics_path = os.path.join(self.temp_dir, "base_metrics.json")
        self.assertTrue(os.path.exists(base_metrics_path), 
                       "base_metrics.json file should be created")
        
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
        report.create_initial_report(X_basic, self.base_metrics, self.temp_dir)
        
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

    def test_create_final_report_workflow(self):
        """Test the complete workflow of creating initial and final reports with degraded data."""
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # Load data and prepare train/test splits (similar to final_report_testing.py)
        df = pd.read_csv("data/WineQT.csv")
        X = df.drop(columns=["quality", "Id"], axis=1)
        X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)
        X_test_splits = np.array_split(X_test, 10)
        
        # Test metrics from final_report_testing.py
        base_metrics = {
            "rmse": 0.6882765026767026,
            "r2": 0.17106058803680269,
            "mae": 0.5595104318406607
        }
        
        # Create initial report
        initial_report_path = os.path.join(self.temp_dir, "initial_report")
        report.create_initial_report(X_train, base_metrics, number_of_output_classes=6, path=initial_report_path)
        
        # Verify initial report files were created
        self.assertTrue(os.path.exists(os.path.join(initial_report_path, "kmeans_clusters.json")), 
                       "kmeans_clusters.json should be created")
        self.assertTrue(os.path.exists(os.path.join(initial_report_path, "base_metrics.json")), 
                       "base_metrics.json should be created")
        self.assertTrue(os.path.exists(os.path.join(initial_report_path, "distribution_descriptors.json")), 
                       "distribution_descriptors.json should be created")
        
        # Load original cluster stats and metrics (mimicking final_report_testing.py)
        with open(os.path.join(initial_report_path, "kmeans_clusters.json"), "r") as f:
            original_cluster_stats = mv.get_cluster_info_from_json(json.load(f))
        
        with open(os.path.join(initial_report_path, "base_metrics.json"), "r") as f:
            loaded_metrics = json.load(f)
        
        # Verify loaded metrics match the original
        self.assertEqual(loaded_metrics, base_metrics, "Loaded metrics should match original base metrics")
        
        # Test metrics evolution data from final_report_testing.py
        new_metrics = [{
            "rmse": 0.6882765026767026,
            "r2": 0.17106058803680269,
            "mae": 0.5595104318406607
        }, {
            "rmse": 0.32,
            "r2": 0.33,
            "mae": 0.45
        }, {
            "rmse": 0.34,
            "r2": 0.2356,
            "mae": 0.23456
        }]
        
        # Create final report
        final_report_path = os.path.join(self.temp_dir, "final_report")
        report.create_report(X_train, original_cluster_stats, X_test_splits, loaded_metrics, final_report_path, new_metrics)
        
        # Verify final report structure was created
        self.assertTrue(os.path.exists(final_report_path), "Final report directory should be created")
        
        # Check that degraded directories were created (one for each split)
        for i in range(len(X_test_splits)):
            degraded_path = os.path.join(final_report_path, f"degraded_{i}")
            self.assertTrue(os.path.exists(degraded_path), 
                           f"degraded_{i} directory should be created")
            
            # Check for distribution comparison file
            comparison_file = os.path.join(degraded_path, f"distribution_comparison_{i}.json")
            self.assertTrue(os.path.exists(comparison_file), 
                           f"distribution_comparison_{i}.json should be created")
        
        # Check evolution directory was created
        evolution_path = os.path.join(final_report_path, "evolution")
        self.assertTrue(os.path.exists(evolution_path), "Evolution directory should be created")
        
        # Check metrics evolution plot was created
        metrics_plot_path = os.path.join(final_report_path, "metrics_evolution.png")
        self.assertTrue(os.path.exists(metrics_plot_path), "metrics_evolution.png should be created")
        
        # Verify cluster comparisons were created
        cluster_path = os.path.join(final_report_path, "clusters")
        self.assertTrue(os.path.exists(cluster_path), "Clusters directory should be created")
        
        for i in range(len(X_test_splits)):
            cluster_comparison_file = os.path.join(cluster_path, f"cluster_comparison_{i}.json")
            self.assertTrue(os.path.exists(cluster_comparison_file), 
                           f"cluster_comparison_{i}.json should be created")
        
        # Verify original cluster stats properties
        self.assertIsInstance(original_cluster_stats.num_clusters, int, 
                             "Number of clusters should be an integer")
        self.assertEqual(original_cluster_stats.num_clusters, 6, 
                        "Number of clusters should be 6 as specified")
        self.assertIsInstance(original_cluster_stats.silhouette_score, (int, float), 
                             "Silhouette score should be numeric")
        
        # Test that new_metrics has the expected structure
        self.assertEqual(len(new_metrics), 3, "Should have 3 metric entries")
        for i, metric_entry in enumerate(new_metrics):
            self.assertIn("rmse", metric_entry, f"Metric entry {i} should have 'rmse'")
            self.assertIn("r2", metric_entry, f"Metric entry {i} should have 'r2'")
            self.assertIn("mae", metric_entry, f"Metric entry {i} should have 'mae'")
            for key, value in metric_entry.items():
                self.assertIsInstance(value, (int, float), 
                                    f"Metric value for {key} should be numeric")

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
