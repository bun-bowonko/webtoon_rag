import pandas as pd

gcs_path = "gs://bun-bucket-test1/data/final_v12.2.1.csv"
df = pd.read_csv(gcs_path)
print(df.head())