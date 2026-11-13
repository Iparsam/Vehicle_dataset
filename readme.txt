# Used Car Price Prediction

Predicting the selling price of used cars with machine learning, from data cleaning and feature engineering to model comparison.

## Overview

The goal is to estimate a used car's selling price from its age, mileage, engine specs, fuel type, transmission and ownership history. The target is trained on a log scale (`log_price`) and predictions are converted back to the original price for evaluation.

## Dataset

- **Source:** [Vehicle dataset from CarDekho (Kaggle)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- **Description:** Listings of used cars from CarDekho, including price, mileage, engine specs and seller details.
- **Size:** 6,698 rows after cleaning
- **Features:** year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats
- **Target:** `selling_price`

## Workflow

1. **Exploratory Data Analysis:** distributions, categorical counts and missing values.
2. **Preprocessing & Encoding:**
   - Parsed the `torque` text column into a numeric value in Nm (kgm converted to Nm).
   - Ordinal encoding for `owner`, binary encoding for `transmission`, one-hot encoding for `fuel` and `seller_type`.
   - Missing values filled with the training-set median.
3. **Feature Engineering:** `car_age`, `km_per_year`, `power_per_cc`, `torque_per_cc`, `log_km_driven`, and `log_price` as the target to reduce skewness.
4. **Modeling & Evaluation:** 80/20 train-test split, five regression models compared with R², MAE and RMSE on the original price scale.

## Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| LightGBM | 0.910 | 72,600 | 152,243 |
| Random Forest | 0.909 | 74,736 | 152,568 |
| XGBoost | 0.904 | 72,724 | 157,157 |
| Ridge | 0.800 | 112,774 | 226,510 |
| Linear Regression | 0.800 | 112,778 | 226,524 |

The three tree-based models perform similarly, with LightGBM slightly ahead. They clearly beat the linear models, which points to non-linear relationships between the features and price.

Most important features: **[feature 1]**, **[feature 2]**, **[feature 3]**.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, LightGBM