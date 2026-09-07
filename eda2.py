import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv("uploads/student_placement_prediction_dataset_2026.csv")

# Create figure
plt.figure(figsize=(8, 5))

# Histogram for Age
plt.hist(
    df["age"],
    bins=10,
    edgecolor="black"
)

# Title
plt.title("Age Distribution")

# X-axis label
plt.xlabel("Age")

# Y-axis label
plt.ylabel("Frequency")

# Grid on Y-axis
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

# Display graph
plt.show()