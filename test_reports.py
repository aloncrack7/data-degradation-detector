from data_degradation_detector import univariate as uv
from data_degradation_detector import multivariate as mv
import data_degradation_detector.report as report
import pandas as pd
import json

df = pd.read_csv("data/WineQT.csv")
X = df.drop(columns=["quality", "Id"], axis=1)

report.create_initial_report(X, "tmp/univariate", number_of_output_classes=2)

descriptors_direct = uv.get_distribution_descriptors_all_columns(X)
json_file = "tmp/univariate/distribution_descriptors.json"
with open(json_file, 'r') as f:
    descriptors_json = json.load(f)
descriptors_json = uv.get_distribution_descriptors_from_json(descriptors_json)

# Check if the descriptors from the DataFrame and JSON match 
print(descriptors_direct == descriptors_json)

df = pd.read_csv("data/basic1.csv")
X = df[["x", "y"]]

report.create_initial_report(X, "tmp/multivariate")

# Check if the number of clusters is 4
with open("tmp/multivariate/kmeans_clusters.json", 'r') as f:
    clustering_data = json.load(f)
    
num_clusters = clustering_data.get('num_clusters', 0)
print(f"Number of clusters: {num_clusters}")
print(f"Number of clusters is 4: {num_clusters == 4}")

clustering_direct = mv.get_cluster_defined_number(X, 4, path="tmp/multivariate2")
descriptors_json = mv.get_cluster_info_from_json(clustering_data)
# Check if the descriptors from the DataFrame and JSON match

print(clustering_direct == descriptors_json)

