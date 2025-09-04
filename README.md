# data-degradation-detector

[![Tests](https://github.com/aloncrack7/data-degradation-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/aloncrack7/data-degradation-detector/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/data-degradation-detector.svg)](https://badge.fury.io/py/data-degradation-detector)
[![Python Support](https://img.shields.io/pypi/pyversions/data-degradation-detector.svg)](https://pypi.org/project/data-degradation-detector/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Library to make estimations on data degradation

## Installation

```bash
pip install data-degradation-detector
```

## Usage

### Basic Example

```python
from data_degradation_detector import univariate, multivariate, report

# Univariate analysis
univariate_results = univariate.analyze('data/WineQT.csv', 'data/WineQT_treated.csv')

# Multivariate analysis
multivariate_results = multivariate.analyze('data/WineQT.csv', 'data/WineQT_treated.csv')

# Generate a report
report.generate(univariate_results, multivariate_results, output_dir='tmp/final_report')
```

### CLI Usage

```bash
python -m data_degradation_detector.univariate data/WineQT.csv data/WineQT_treated.csv
python -m data_degradation_detector.multivariate data/WineQT.csv data/WineQT_treated.csv
```


## API Reference

### `univariate` module

- `get_distribution_descriptors(column: pd.Series) -> DistributionDescriptors`  
	Compute distribution descriptors (mean, std, min, max, quartiles) for a column.
- `get_distribution_descriptors_all_columns(df: pd.DataFrame) -> dict`  
	Compute descriptors for all columns in a DataFrame.
- `plot_distribution_descriptors(column: pd.Series, ...)`  
	Plot distribution and descriptors for a column.
- `plot_distribution_descriptors_all_columns(df: pd.DataFrame, ...)`  
	Plot distributions for all columns.
- `compare_distributions(original: pd.Series, new_data: pd.Series, ...) -> DistributionChanges`  
	Compare two distributions and visualize changes.
- `compare_distribbutions_all_columns(original: pd.DataFrame, new_data: pd.DataFrame, ...)`  
	Compare all columns between two DataFrames.
- `descriptor_evolution(dfs: list[pd.Series], ...)`  
	Plot evolution of descriptors for a column across multiple DataFrames.
- `descriptor_evolution_all_columns(dfs: list[pd.DataFrame], ...)`  
	Plot evolution for all columns.

### `multivariate` module

- `get_best_clusters(X: pd.DataFrame, path: str = None, plot: bool = True) -> Cluster_statistics`  
	Find optimal KMeans clusters and return statistics.
- `get_cluster_defined_number(X: pd.DataFrame, num_clusters: int, ...) -> Cluster_statistics`  
	Run KMeans with a fixed number of clusters.
- `compare_clusters(cluster_stats1, cluster_stats2, delta=0.1) -> ClusterChanges`  
	Compare two clusterings and return changes.
- `clustering_evolution(dfs: list[pd.DataFrame], num_clusters: int, ...)`  
	Visualize clustering evolution across multiple DataFrames.
- `correlation_matrix(df: pd.DataFrame, path: str = None)`  
	Plot and/or save a correlation matrix heatmap.
- `get_cluster_info_from_json(json_data)`  
	Load cluster statistics from JSON.

#### Classes
- `Cluster_statistics`  
	Holds statistics for a clustering (num_clusters, silhouette, centroids, radius, label percentages).
- `ClusterChanges`  
	Represents and quantifies changes between two clusterings.

### `report` module

- `get_number_of_output_classes(y: pd.Series) -> int`  
	Get the number of unique classes in a target column.
- `create_initial_report(df: pd.DataFrame, target: str, base_metrics: dict, path: str, number_of_output_classes: int = None)`  
	Generate initial visualizations and statistics for a dataset.
- `create_report(original_df, original_clusters, degraded_dfs, base_metrics, path, new_metrics=None)`  
	Generate a full report comparing original and degraded datasets.

---
For more details, see the source code or the [documentation](https://github.com/aloncrack7/data-degradation-detector).

## Development

### Running Tests

```bash
python -m pytest testing/ -v
```

### Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/aloncrack7/data-degradation-detector).

## License

This project is licensed under the GPL v3 License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, open an issue on [GitHub](https://github.com/aloncrack7/data-degradation-detector/issues).
