import pandas as pd
from . import univariate as uv
from . import multivariate as mv
import json
import os

def create_initial_report(df: pd.DataFrame, path: str, number_of_output_classes: int = None) -> None:
    """
    Create the initial informative visualizations and statistics for the given DataFrame.
    """
    # Get distribution descriptors for all columns
    descriptors = uv.get_distribution_descriptors_all_columns(df)
    descriptors = {k: v.get_json() for k, v in descriptors.items()}

    os.makedirs(path, exist_ok=True)
    with open(f"{path}/distribution_descriptors.json", 'w+') as f:
        json.dump(descriptors, f, indent=4)

    # Plot distribution descriptors for all columns
    uv.plot_distribution_descriptors_all_columns(df, path=path)

    if number_of_output_classes is not None:
        cluster_info = mv.get_cluster_defined_number(df, number_of_output_classes, path=path)
    else:
        cluster_info = mv.get_best_clusters(df, path=path)

    with open(f"{path}/kmeans_clusters.json", 'w+') as f:
        json.dump(cluster_info.get_json(), f, indent=4)