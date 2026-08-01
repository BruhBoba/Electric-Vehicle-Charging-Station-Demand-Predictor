import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import prepare_data 
from model import train_models

x_train_final, x_test_final, y_train, y_test = prepare_data()
rf_model, rf_prediction, lr_model, lr_prediction = train_models()

rf_mae = mean_absolute_error(y_test, rf_prediction)
lr_mae = mean_absolute_error(y_test, lr_prediction)

print("MODEL REPORT CARD:")
print("-------- Random Forest --------")
print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")  
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, rf_prediction)):.4f}")
print(f"R-squared (R²): {r2_score(y_test, rf_prediction):.4F}")      


print("\n-------- Linear Regression --------")
print(f"Mean Absolute Error (MAE): {lr_mae:.4f}")  
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, lr_prediction)):.4f}")
print(f"R-squared (R²): {r2_score(y_test, lr_prediction):.4F}")      

improvement = ((lr_mae - rf_mae) / lr_mae) * 100
print(f"Random Forest reduced prediction error by {improvement:.2f}%")

"""
Looking at the results for Random forest the MAE is 0.0724, RMSE is 0.1201, and R²(accuracy) is 0.8682. For Linear 
Regression, the MAE is 0.2204, RMSE is 0.2666, and R² is 0.3506. Which concludes that Random Forest 
overperformed Linear Regression because it shows the relationships in EV charging demand more effectively. 
"""

# Model overfitting 
train_pred = rf_model.predict(x_train_final)
test_pred = rf_model.predict(x_test_final)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("\nModel Overfitting Test:")
print(f"Training R²: {train_r2:.4f}") 
print(f"Testing R² : {test_r2:.4f}")  
# Mild sign of overfitting, training R² = 0.9168 and testing R² = 0.8682. 
# The gap is about 0.05 (or 4.86%), which is small


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
# X value of 0.05 - 0.2, more points are above the line which model predicted demand higher than actual
# Most predictions closely follow the ideal diagonal line, prediction errors become slightly larger 
# during very high utilization periods


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
# Most residuals are centered around zero which is a good but prediction errors increase for 
# higher utilization values.