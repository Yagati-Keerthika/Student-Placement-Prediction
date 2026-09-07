import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
file_path = "uploads/student_placement_prediction_dataset_2026.csv"

df = pd.read_csv(file_path)

# Create figure
plt.figure(figsize=(8, 5))

# Count plot for branch
sns.countplot(
    x=df["branch"]
)

# Title
plt.title("Students by Branch")

# X-axis
plt.xlabel("Branch")

# Y-axis
plt.ylabel("Number of Students")

# Display graph
plt.show()