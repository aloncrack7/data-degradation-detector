from data_degradation_detector import univariate as uv
import data_degradation_detector.report as report
import pandas as pd
import json

df = pd.read_csv("data/WineQT.csv")
X = df.drop(columns=["quality", "Id"], axis=1)

report.create_initial_inform(X, "tmp")

descriptors_direct = uv.get_distribution_descriptors_all_columns(X)
json_file = "tmp/distribution_descriptors.json"
with open(json_file, 'r') as f:
    descriptors_json = json.load(f)
descriptors_json = uv.get_distribution_descriptors_from_json(descriptors_json)

# Check if the descriptors from the DataFrame and JSON match 
print(descriptors_direct == descriptors_json)

