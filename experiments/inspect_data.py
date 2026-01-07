
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/PartC/training_data.csv")
print(df.describe())

# Histogram of Cd
plt.figure()
df['cd'].hist(bins=50, range=(0, 0.5))
plt.title("Cd Distribution")
plt.savefig("data/PartC/figures/cd_dist.png")
