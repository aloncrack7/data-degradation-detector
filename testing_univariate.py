# %% [markdown]
# # Testing

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
from data_degradation_detector import univariate as uv
import matplotlib.pyplot as plt
import numpy as np
import importlib
importlib.reload(uv)


# %%
df = pd.read_csv("data/WineQT.csv")
df = df.drop(columns=['quality', 'Id'], axis=1)
df

# %%
print(uv.get_distribution_descriptors_all_columns(df))

# %%
uv.plot_distribution_descriptors_all_columns(df)

# %%
df1 = df.iloc[:len(df)//2].reset_index(drop=True)
df2 = df.iloc[len(df)//2:].reset_index(drop=True)

uv.compare_distribbutions_all_columns(df1, df2)

# %%
dfs = np.array_split(df, 10)
uv.descriptor_evolution_all_columns(dfs)


