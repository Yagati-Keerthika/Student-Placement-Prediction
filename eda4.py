import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
file_path = "uploads/student_placement_prediction_dataset_2026.csv"

df = pd.read_csv(file_path)

# Create box plot
plt.figure(figsize=(6, 4))

plt.boxplot(
    df["age"].dropna()
)

plt.title("Box Plot of Age")
plt.ylabel("Age")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.show()