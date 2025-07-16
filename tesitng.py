import pandas as pd
from data_degradation_detector import univariate as uv
import numpy as np

df = pd.read_csv("WineQT.csv")

dfs = np.array_split(df, 10)
uv.descriptor_evolution_from_dfs(dfs, 'fixed acidity')