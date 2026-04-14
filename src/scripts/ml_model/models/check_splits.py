from pathlib import Path
import pandas as pd

base = Path("/home/ellab/air_pollution/src/data/ml_data/splits_3way")

train = pd.read_parquet(base / "train.parquet")
val   = pd.read_parquet(base / "val.parquet")
test  = pd.read_parquet(base / "test.parquet")

# 1) Ensure no sites overlap
train_sites = set(train["site_id"].unique())
val_sites   = set(val["site_id"].unique())
test_sites  = set(test["site_id"].unique())

print("Overlap train/val:", len(train_sites & val_sites))
print("Overlap train/test:", len(train_sites & test_sites))
print("Overlap val/test:", len(val_sites & test_sites))

# 2) Check n_days rule is enforced
print("Min n_days train:", train["n_days"].min())
print("Min n_days val  :", val["n_days"].min())
print("Min n_days test :", test["n_days"].min())
