from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from features import prepare_data 

x_train_final, x_test_final, y_train, y_test = prepare_data()

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=117)
# train model
rf_model.fit(x_train_final, y_train)
rf_prediction = rf_model.predict(x_test_final)


# Linear Regression model 
lr_model = LinearRegression()
lr_model.fit(x_train_final, y_train)
lr_prediction = lr_model.predict(x_test_final)


# Naive baseline — no model, no training
# guess = average utilization for that hour of day, learned from TRAIN only
hourly_mean = y_train.groupby(x_train_final['hour_of_day']).mean()
baseline_prediction = x_test_final['hour_of_day'].map(hourly_mean)

def train_models():
    return (
        rf_model,
        rf_prediction,
        lr_model,
        lr_prediction,
        baseline_prediction
    )
