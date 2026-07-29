from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import prepare_data 
from model import train_models

x_train_final, x_test_final, y_train, y_test = prepare_data()
rf_model, y_prediction = train_models()

mae = mean_absolute_error(y_test, y_prediction)
# the function below is resulting in errors, lowk tis discontinued
#rmse = mean_squared_error(y_test, y_prediction, squared=False) 
r2 = r2_score(y_test, y_prediction)

print("--- MODEL REPORT CARD ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")  #0.0753
#print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")      #0.8534