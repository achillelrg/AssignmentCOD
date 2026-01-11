"""
Data Inspection Utility.

This script loads the generated training data (`data/PartC/training_data.csv`)
and plots the coverage/distribution of aerodynamic coefficients (Cl, Cd, Cm).
Used to verify that LHS sampled a diverse range of physical behaviors.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/PartC/training_data.csv")
print(df.describe())

# Histogram of Cd
plt.figure()
df['cd'].hist(bins=50, range=(0, 0.5), color='blue', alpha=0.7)
plt.title("Cd Distribution")
plt.xlabel("Cd")
plt.ylabel("Frequency")
plt.savefig("data/PartC/figures/cd_dist.png")
plt.close()

# Histogram of Cl
plt.figure()
df['cl'].hist(bins=50, color='green', alpha=0.7)
plt.title("Cl Distribution")
plt.xlabel("Cl")
plt.ylabel("Frequency")
plt.savefig("data/PartC/figures/cl_dist.png")
plt.close()

# Histogram of Cm
plt.figure()
df['cm'].hist(bins=50, color='red', alpha=0.7)
plt.title("Cm Distribution")
plt.xlabel("Cm")
plt.ylabel("Frequency")
plt.savefig("data/PartC/figures/cm_dist.png")
plt.close()
