import pandas as pd

CSV_PATH = r"C:\Users\siva\Desktop\VIT\CNN\sentinel\tirunelveli_rf_36_features.csv"

df = pd.read_csv(CSV_PATH)

print("\n==============================")
print("CSV CHECK")
print("==============================")

print("\nShape:")
print(df.shape)

print("\nColumns:")
for i, col in enumerate(df.columns):
    print(i, ":", col)

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
if "class" in df.columns:
    print(df["class"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate columns:")
print(df.columns[df.columns.duplicated()].tolist())