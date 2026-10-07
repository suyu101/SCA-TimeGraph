import pandas as pd

file = "Datasets/A1/Gaussian/4 variable/Lag 2/linear_ts_n500_vars4_lag2.csv"

df = pd.read_csv(file)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df[["X1", "X2", "X3", "X4"]].describe())
