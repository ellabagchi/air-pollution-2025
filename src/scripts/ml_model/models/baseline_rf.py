from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


base = Path("/home/ellab/air_pollution/src/data/ml_data/splits_3way")

train = pd.read_parquet(base / "train.parquet")
val   = pd.read_parquet(base / "val.parquet")
test  = pd.read_parquet(base / "test.parquet")

# Features (simple baseline)
feature_cols = ["Latitude", "Longitude", "year", "month"]

X_train = train[feature_cols]
y_train = train["pm25_monthly_mean"]

X_val = val[feature_cols]
y_val = val["pm25_monthly_mean"]

X_test = test[feature_cols]
y_test = test["pm25_monthly_mean"]

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

def report(split_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{split_name}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("Baseline Random Forest")
report("VAL ", y_val, val_pred)
report("TEST", y_test, test_pred)
