import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import prepare_data 
from model import train_models

x_train_final, x_test_final, y_train, y_test = prepare_data()
rf_model, rf_prediction, lr_model, lr_prediction, baseline_prediction = train_models()

# Baseline shouldn't have any unmapped hours
print(f"Baseline NaN count: {baseline_prediction.isna().sum()}")  # equals to 0

rf_mae = mean_absolute_error(y_test, rf_prediction)
lr_mae = mean_absolute_error(y_test, lr_prediction)
baseline_mae = mean_absolute_error(y_test, baseline_prediction)

print("MODEL REPORT CARD:")
print("-------- Random Forest --------")
print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")  
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, rf_prediction)):.4f}")
print(f"R-squared (R²): {r2_score(y_test, rf_prediction):.4F}")      

print("\n-------- Linear Regression --------")
print(f"Mean Absolute Error (MAE): {lr_mae:.4f}")  
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, lr_prediction)):.4f}")
print(f"R-squared (R²): {r2_score(y_test, lr_prediction):.4F}")      

print("\n-------- Naive Baseline (avg utilization per hour) --------")
print(f"Mean Absolute Error (MAE): {baseline_mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, baseline_prediction)):.4f}")
print(f"R-squared (R²): {r2_score(y_test, baseline_prediction):.4f}")

improvement_lr_vs_baseline = ((baseline_mae - lr_mae) / baseline_mae) * 100
improvement_rf_vs_lr = ((lr_mae - rf_mae) / lr_mae) * 100
improvement_rf_vs_baseline = ((baseline_mae - rf_mae) / baseline_mae) * 100

print(f"\nMoving from a naive guess to Linear Regression cut prediction error by {improvement_lr_vs_baseline:.2f}%.")
print(f"Moving from Linear Regression to Random Forest cut prediction error by {improvement_rf_vs_lr:.2f}%.")
print(f"Overall, Random Forest cut prediction error by {improvement_rf_vs_baseline:.2f}% compared to guessing alone.")

"""
Looking at the results for Random forest the MAE is 0.0582, RMSE is 0.0844, and R²(accuracy) is 0.9399. For Linear 
Regression, the MAE is 0.0782, RMSE is 0.1078, and R² is 0.9020. For the naive baseline, the MAE is 0.2025, RMSE is
0.2576, and R² is 0.4401. Which concludes that Random Forest overperformed Linear Regression and naive baseline
because it shows the relationships in EV charging demand more effectively. naive baseline < Linear Regression < Random Forest
"""

# Model overfitting 
train_pred = rf_model.predict(x_train_final)
test_pred = rf_model.predict(x_test_final)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("\nModel Overfitting Test:")
print(f"Training R²: {train_r2:.4f}") 
print(f"Testing R² : {test_r2:.4f}")  
# There is no overfitting, training R² = 0.9670 and testing R² = 0.9399. 
# Both values are high and very close with a difference of 0.0271 (2.71%).


# Feature importance: evaluating and interpreting the model to show the strongest predictor of demand 
importance = pd.DataFrame({
    "Feature": x_train_final.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print(importance.head(10))

# Graph of the feature importance 
plt.figure(figsize=(8,6))
plt.barh(
    importance["Feature"][:10],
    importance["Importance"][:10]
)
plt.title("Top 10 Most Important Features")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()


# Actual vs. Predicted Graph 
plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    rf_prediction,
    alpha=.5
)

plt.plot(
    [0,1],
    [0,1],
    color="red"
)

plt.xlabel("Actual Utilization")
plt.ylabel("Predicted Utilization")
plt.title("Actual vs Predicted")
plt.show()
# Most predictions closely follow the ideal diagonal line, although there are some predictions (0.0-0.2) that are
# above the diagonal line where the model is overpredicting when true demand is low and util rate (0.8-1.0) 
# where the model is underpredicting when true demand is high. The model is most accurate in the mid-range.  


# Residual analysis graph
# Residual = Actual - Predicted so shows the prediction error 
residuals = y_test - rf_prediction
plt.figure(figsize=(7,6))

plt.scatter(
    rf_prediction,
    residuals,
    alpha=.5
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Utilization Rate")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()
# Most residuals are centered around zero which is a good but for util rates from 0.4-0.6, prediction errors increase
# for higher utilization values.


# Error breakdown by segment
# Build a results dataframe aligned to the test set
results = pd.DataFrame({
    "actual": y_test,
    "predicted": rf_prediction,
    "error": y_test - rf_prediction,          # positive = underprediction, negative = overprediction
    "abs_error": np.abs(y_test - rf_prediction),
    "hour_of_day": x_test_final["hour_of_day"].values
})

# --- By hour of day ---
hourly_error = results.groupby("hour_of_day").agg(
    mae=("abs_error", "mean"),
    mean_error=("error", "mean"),   
    n=("error", "size")
).sort_values("mae", ascending=False)

print("\n-------- Error by Hour of Day --------")
print(hourly_error)

results["util_bucket"] = pd.cut(
    results["actual"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
)

bucket_error = results.groupby("util_bucket", observed=True).agg(
    mae=("abs_error", "mean"),
    mean_error=("error", "mean"),
    n=("error", "size")
)

print("\n-------- Error by Actual Utilization Range --------")
print(bucket_error)

# Bar chart
plt.figure(figsize=(8,5))
plt.bar(bucket_error.index.astype(str), bucket_error["mean_error"])
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Actual Utilization Range")
plt.ylabel("Mean Error (Actual - Predicted)")
plt.title("Prediction Bias by Utilization Range")
plt.show()
# Positive bars = model underpredicts in that range
# Negative bars = model overpredicts in that range
# Model underpredicts high-demand periods for 0.8–1.0 utilization by an average of 0.098 bias and for 0.4–0.6 utilization
# it overpredicts. 


# Saving the Random Forest model 
joblib.dump(rf_model, "models/random_forest.joblib")
joblib.dump(list(x_train_final.columns), "models/features.joblib")

print("Model saved successfully!")