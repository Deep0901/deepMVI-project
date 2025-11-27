import pandas as pd
import numpy as np
import os

src = "/mnt/data/electricity.txt"   # file you uploaded earlier
dst = "/home/deeps/deepmvi-seminar/data/electricity/X.npy"

print("Loading:", src)
df = pd.read_csv(src, sep=None, engine="python")

# Drop non-numeric columns
for col in df.columns:
    if not np.issubdtype(df[col].dtype, np.number):
        df = df.drop(columns=[col])

X = df.values.astype(float)
os.makedirs("/home/deeps/deepmvi-seminar/data/electricity", exist_ok=True)
np.save(dst, X)

print("Saved electricity X.npy to:", dst)
print("Shape:", X.shape)
