import pandas as pd
from . import univariate as uv
import json
import os

def create_initial_inform(df: pd.DataFrame, path: str) -> None:
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