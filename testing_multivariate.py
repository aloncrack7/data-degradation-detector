# %% [markdown]
# # Testing multivariate

# %% [markdown]
# ## Import libraries

# %%
from data_degradation_detector import multivariate as mv
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import numpy as np
importlib.reload(mv)

# %% [markdown]
# ## Import data

# %%
df = pd.read_csv("data/basic1.csv")
X = df.drop(columns=['color'], axis=1)
y = df['color']

# %%
X

# %%
y

# %% [markdown]
# ## Test the module

# %% [markdown]
# ### Clustering

# %%
mv.get_best_clusters(X)

# %%
mv.get_cluster_defined_number(X, 10)

# %%
mid = len(X) // 2
X_train, X_test = X.iloc[:mid], X.iloc[mid:]
y_train, y_test = y.iloc[:mid], y.iloc[mid:]

# %%
bc1 = mv.get_best_clusters(X_train)
bc2 = mv.get_best_clusters(X_test)

mv.compare_clusters(bc1, bc2)

# %%
bc3 = mv.get_cluster_defined_number(X_train, 2)
bc4 = mv.get_cluster_defined_number(X_test, 2)

mv.compare_clusters(bc3, bc4)

# %%
Xs = np.array_split(X, 10)
mv.clustering_evolution(Xs, 4)


