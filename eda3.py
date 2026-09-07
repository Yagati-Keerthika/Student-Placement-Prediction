import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
file_path = "uploads/student_placement_prediction_dataset_2026.csv"

df = pd.read_csv(file_path)

# Create figure
plt.figure(figsize=(8, 5))

# Histogram for CGPA
plt.hist(
    df["cgpa"],
    bins=10,
    edgecolor="black"
)

# Title
plt.title("CGPA Distribution")

# X-axis label
plt.xlabel("CGPA")

# Y-axis label
plt.ylabel("Frequency")

# Grid
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

# Display graph
plt.show()