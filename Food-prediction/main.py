#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Use paths relative to this script, so it works regardless of
# where you run "streamlit run".
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "Food delivery.csv"
SCALER_PATH = BASE_DIR / "standard_scaler.pkl"
MODEL_PATH = BASE_DIR / "xgb_best.pkl"

df = pd.read_csv(CSV_PATH)

df.head()
df.describe()
df.info()

type(df["Delivery_person_Age"][0])

df["Weatherconditions"].value_counts()

df["Weatherconditions"] = df["Weatherconditions"].apply(
    lambda temp: temp.split(" ")[1] if pd.notnull(temp) else temp
)

df["Weatherconditions"]

df.columns

df.select_dtypes(include="object").columns

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Standardize common textual missing-value markers
missing_values = ["NaN", "nan", "NAN", "NA", "N/A", "null", "NULL", "None", ""]

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(missing_values, np.nan)

df["Delivery_person_Age"].value_counts()

df["Time_taken(min)"]

df["Time_taken(min)"] = df["Time_taken(min)"].apply(
    lambda temp: int(temp.split(" ")[1]) if pd.notnull(temp) else temp
)

df["Time_taken(min)"]

df.head()

for col in [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.info()

df[df["Delivery_person_Ratings"] > 5]["Delivery_person_Ratings"].shape

df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(upper=5)

df.head()
df.describe()

# Numeric columns: fill missing values with the median
for col in [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
]:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: fill missing values with the mode
for col in ["Weatherconditions", "Road_traffic_density", "Festival", "City"]:
    df[col] = df[col].fillna(df[col].mode()[0])

df.info()

df["Road_traffic_density"].value_counts()

sns.histplot(df["Time_taken(min)"], bins=30, kde=True)
plt.title("Distribution of delivery time")
plt.show()

df.info()

df.groupby("Road_traffic_density")["Time_taken(min)"].agg(
    ["count", "mean", "median", "std"]
)

df.groupby("Weatherconditions")["Time_taken(min)"].agg(
    ["count", "mean", "median", "std"]
)

df.groupby("Festival")["Time_taken(min)"].agg(["count", "mean", "median", "std"])

df.groupby("City")["Time_taken(min)"].agg(["count", "mean"])

df.groupby("Type_of_vehicle")["Time_taken(min)"].agg(["count", "mean"])

df.groupby("Type_of_order")["Time_taken(min)"].agg(["count", "mean"])

df["Delivery_person_Age"].describe()

sns.histplot(df["Delivery_person_Age"], bins=10)

df[df["Delivery_person_Age"] > 40]["Delivery_person_Age"].shape

plt.figure(figsize=(10, 5))

sns.lineplot(
    data=df,
    x="Delivery_person_Age",
    y="Time_taken(min)",
    estimator="mean",
    errorbar=None,
    marker="o",
)

plt.title("Average Delivery Time vs Delivery Person Age")
plt.xlabel("Delivery Person Age")
plt.ylabel("Average Delivery Time (minutes)")
plt.show()

df["Delivery_person_Ratings"].describe()

plt.figure(figsize=(10, 5))

sns.lineplot(
    data=df,
    x="Delivery_person_Ratings",
    y="Time_taken(min)",
    estimator="mean",
    errorbar=None,
    marker="o",
)

plt.title("Average Delivery Time vs Delivery Person Rating")
plt.xlabel("Delivery Person Rating")
plt.ylabel("Average Delivery Time (minutes)")
plt.show()

df[df["Delivery_person_Ratings"] < 4].shape[0] / len(df)

numeric_cols = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
    "Vehicle_condition",
    "Time_taken(min)",
]

sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation — numeric columns before feature engineering")
plt.show()

df.info()

df["Order_Date"]
df["Time_Orderd"]

hour = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce").dt.hour

hour.unique()

df.groupby(hour)["Time_taken(min)"].agg(["mean", "median"])

df.groupby(hour)["Time_taken(min)"].mean().plot(kind="bar", figsize=(11, 4))
plt.title("Average Delivery Time by Order Hour")
plt.ylabel("Avg minutes")
plt.show()

df.info()

# Distance from coordinates


def calculate_series_distance(
    lat1_series, lon1_series, lat2_series, lon2_series, miles=False
):
    """Calculate great-circle distance using the Haversine formula."""
    r = 3956 if miles else 6371

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1_series, lon1_series, lat2_series, lon2_series],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * r


df["distance_km"] = calculate_series_distance(
    df["Restaurant_latitude"],
    df["Restaurant_longitude"],
    df["Delivery_location_latitude"],
    df["Delivery_location_longitude"],
)

df.info()
df["distance_km"].describe()

df[df["distance_km"] > 100]["distance_km"].describe()
df[df["distance_km"] < 100]["distance_km"].describe()
df[df["distance_km"] > 100]

df = df[df["distance_km"] < 100].reset_index(drop=True)

df.info()

df["order_hour"] = pd.to_datetime(
    df["Time_Orderd"], format="%H:%M:%S", errors="coerce"
).dt.hour

df["order_hour"] = df["order_hour"].fillna(df["order_hour"].median())

df.info()

len(df["Delivery_person_ID"].unique())

features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_km",
    "order_hour",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
]

X = df[features]
y = df["Time_taken(min)"]

# Encoding

traffic_mapping = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
X["Road_traffic_density"] = X["Road_traffic_density"].map(traffic_mapping)

X["Road_traffic_density"]

X["Weatherconditions"].value_counts()

X = pd.get_dummies(
    X,
    columns=[
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
        "Weatherconditions",
    ],
    drop_first=True,
    dtype=int,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

X_train.shape, X_test.shape

# Feature scaling

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index,
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index,
)

# Save the fitted scaler beside main.py
joblib.dump(scaler, SCALER_PATH)

# Linear regression pipeline

lr = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

coef_df = pd.DataFrame(
    {
        "feature": X_train.columns,
        "coefficient": lr.named_steps["regressor"].coef_,
    }
).sort_values("coefficient", key=abs, ascending=False)

coef_df
X_train.columns

# Linear regression evaluation

y_pred_test = lr.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

y_pred_train = lr.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

print("Linear regression train RMSE:", rmse_train)
print("Linear regression train R2:", r2_train)
print("Linear regression test RMSE:", rmse_test)
print("Linear regression test R2:", r2_test)

# Decision tree

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)

dt_test_pred = dt.predict(X_test_scaled)
dt_train_pred = dt.predict(X_train_scaled)

print("Decision tree test RMSE:", np.sqrt(mean_squared_error(y_test, dt_test_pred)))
print("Decision tree test R2:", r2_score(y_test, dt_test_pred))

print("Decision tree train RMSE:", np.sqrt(mean_squared_error(y_train, dt_train_pred)))
print("Decision tree train R2:", r2_score(y_train, dt_train_pred))

# Random forest

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

rf_test_pred = rf.predict(X_test_scaled)
rf_train_pred = rf.predict(X_train_scaled)

print("Random forest test RMSE:", np.sqrt(mean_squared_error(y_test, rf_test_pred)))
print("Random forest test R2:", r2_score(y_test, rf_test_pred))

print("Random forest train RMSE:", np.sqrt(mean_squared_error(y_train, rf_train_pred)))
print("Random forest train R2:", r2_score(y_train, rf_train_pred))

# Random forest tuning

rf_param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [10, 15, 20, 25, None],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_dist,
    n_iter=25,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
)

rf_search.fit(X_train_scaled, y_train)

print("Best random forest params:", rf_search.best_params_)

rf_best = rf_search.best_estimator_

rf_best_test_pred = rf_best.predict(X_test_scaled)

r2_rf_tuned = r2_score(y_test, rf_best_test_pred)
rmse_rf_tuned = np.sqrt(mean_squared_error(y_test, rf_best_test_pred))

print("Tuned random forest RMSE:", rmse_rf_tuned)
print("Tuned random forest R2:", r2_rf_tuned)

rf.feature_importances_
X_train.columns

# XGBoost

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_scaled, y_train)

xgb_test_pred = xgb.predict(X_test_scaled)

print("XGBoost test RMSE:", np.sqrt(mean_squared_error(y_test, xgb_test_pred)))
print("XGBoost test R2:", r2_score(y_test, xgb_test_pred))

# XGBoost tuning

xgb_param_dist = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [1, 1.5, 2],
}

xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    xgb_param_dist,
    n_iter=40,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
)

xgb_search.fit(X_train_scaled, y_train)

print("Best XGBoost params:", xgb_search.best_params_)

xgb_best = xgb_search.best_estimator_

xgb_train_pred = xgb_best.predict(X_train_scaled)
xgb_test_pred = xgb_best.predict(X_test_scaled)

r2_xgb_train = r2_score(y_train, xgb_train_pred)
rmse_xgb_train = np.sqrt(mean_squared_error(y_train, xgb_train_pred))

r2_xgb_test = r2_score(y_test, xgb_test_pred)
rmse_xgb_test = np.sqrt(mean_squared_error(y_test, xgb_test_pred))

print("XGBoost train RMSE:", rmse_xgb_train)
print("XGBoost train R2:", r2_xgb_train)
print("XGBoost test RMSE:", rmse_xgb_test)
print("XGBoost test R2:", r2_xgb_test)

X_test_scaled.shape[1]

# Adjusted R2

n_test = len(y_test)
p_test = X_test_scaled.shape[1]

adj_r2_xgb_test = 1 - (1 - r2_xgb_test) * (n_test - 1) / (n_test - p_test - 1)

print("XGBoost test adjusted R2:", adj_r2_xgb_test)

# Order period feature


def order_period(hour):
    if 12 <= hour <= 14:
        return "Lunch"
    elif 19 <= hour <= 21:
        return "Dinner"
    return "Normal"


X_train_exp = X_train.copy()
X_test_exp = X_test.copy()

X_train_exp["order_period"] = X_train_exp["order_hour"].apply(order_period)
X_test_exp["order_period"] = X_test_exp["order_hour"].apply(order_period)

X_train_exp = pd.get_dummies(
    X_train_exp,
    columns=["order_period"],
    drop_first=True,
    dtype=int,
)

X_test_exp = pd.get_dummies(
    X_test_exp,
    columns=["order_period"],
    drop_first=True,
    dtype=int,
)

# Align columns in case a category appears in train but not test
X_train_exp, X_test_exp = X_train_exp.align(
    X_test_exp,
    join="outer",
    axis=1,
    fill_value=0,
)

scaler_exp = StandardScaler()

X_train_exp_scaled = scaler_exp.fit_transform(X_train_exp)
X_test_exp_scaled = scaler_exp.transform(X_test_exp)

xgb_with_period = XGBRegressor(
    **xgb_search.best_params_,
    random_state=42,
)

xgb_with_period.fit(X_train_exp_scaled, y_train)

r2_with_period_test = r2_score(
    y_test,
    xgb_with_period.predict(X_test_exp_scaled),
)

n_test = X_test_exp.shape[0]
p_test = X_test_exp.shape[1]

adj_r2_test = 1 - (1 - r2_with_period_test) * (n_test - 1) / (n_test - p_test - 1)

print("Test adjusted R2 with order period:", adj_r2_test)

# Save model beside main.py
joblib.dump(xgb_best, MODEL_PATH)

# Load saved artifacts using script-relative paths
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

sample = X_test[0:3]
sample_scaled = scaler.transform(sample)

predictions = model.predict(sample_scaled)

predictions
y_test[0:3].values
