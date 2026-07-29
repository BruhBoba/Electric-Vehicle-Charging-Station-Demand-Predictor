from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from features import prepare_data 

x_train_final, x_test_final, y_train, y_test = prepare_data()

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=117)

# train model
rf_model.fit(x_train_final, y_train)

y_prediction = rf_model.predict(x_test_final)


# Linear Regression model 
# It is a baseline model to compare against Random Forest to see if it's the best model
lr_model = LinearRegression()

lr_model.fit(x_train_final, y_train)

lr_predictions = lr_model.predict(x_test_final)

def train_models():
    return (
        rf_model,
        y_prediction,
        lr_model,
        lr_predictions
    )