import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
file_path = "uploads/student_placement_prediction_dataset_2026.csv"

df = pd.read_csv(file_path)

# Number of rows and columns
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# First 10 rows
print("\nFirst 10 Rows:")
print(df.head(10))

# Last 10 rows
print("\nLast 10 Rows:")
print(df.tail(10))

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data types and memory information
print("\nData Information:")
df.info()

# Statistical information
print("\nStatistical Information:")
print(df.describe())

# Numerical columns
numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)

# Categorical columns
categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)