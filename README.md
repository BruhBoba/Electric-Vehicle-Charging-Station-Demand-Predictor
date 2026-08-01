# Electric Vehicle Charging Station Demand Predictor

## Overview

This project predicts electric vehicle (EV) charging station utilization 30 minutes into the future using machine learning. By forecasting short-term charging demand, the model can help improve charger availability, reduce waiting times, better load balancing, and support smarter charging infrastructure planning.


## Research Question

How accurately can machine learning forecast EV charging station utilization 30 minutes in advance in Austin, TX?


## Dataset

The dataset used in this project comes from Kaggle:

**EV Charging Station Availability Tracking**

https://www.kaggle.com/datasets/likithagedipudi/ev-charging-station-availability-tracking

The dataset contains information including:
- Charging station information
- Charger type
- Network
- Amenities nearby
- Traffic congestion
- Weather conditions
- Local events
- Power output
- Historical utilization rates
- Pricing
- Gas price per gallon
- Time and date information


## Project Workflow

The project follows a complete machine learning pipeline:

1. Data preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Feature selection using Pearson and Spearman correlation
5. Time-series train/test split
6. Random Forest Regression
7. Linear Regression model comparison
8. Naive baseline 
9. Model evaluation
10. Feature importance analysis
11. Residual analysis
12. Error breakdown by segment


## Repository Structure

```
project/
│
├── data/
│   └── austin_ev.csv
│
├── source/
│   ├── preprocess.py
│   ├── ev_data_eda.py
│   ├── features.py
│   ├── model.py
│   └── model_evaluation.py
│   ├── pearson
│   └── spearman
│
├── ev_charging_data_visualization.ipynb
├── requirements.txt
└── README.md
```


## Installation

Clone this repository:

```bash
git clone <repository-url>
```

Move into the project folder:

```bash
cd Electric-Vehicle-Charging-Station-Demand-Predictor
```

Install the required packages:

```bash
pip install -r requirements.txt
```


## Required Libraries

The required third-party packages are listed in requirements.txt.

They include:
- scikit-learn
- pandas 
- numpy 
- matplotlib
- seaborn


## Running the Project

Run each stage of the pipeline in order.

### 1. Preprocess Dataset 
```bash
python source/preprocess.py
```
This script:
- filters the dataset to Austin, TX only

### 2. Data Preprocessing / EDA 
```bash
python source/ev_data_eda.py
```
This script:
- loads the austin_ev dataset
- checks for missing values
- checks for duplicate rows 
- checks if utilization rate is within range
- checks unique values in categorical columns

### 3. Feature Engineering
```bash
python source/features.py
```

This script:
- loads the dataset
- creates lag features
- creates the prediction target
- removes leakage features
- performs feature selection
- performs Pearson and Spearman correlation analysis
- prepares Time Series training and testing datasets

### 4. Train the Models
```bash
python source/model.py
```

This script trains:
- Random Forest Regressor
- Linear Regression
- Naive Baseline 

### 5. Evaluate Performance
```bash
python source/model_evaluation.py
```

The evaluation includes:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Training vs Testing performance
- Residual analysis
- Actual vs Predicted plots
- Random Forest feature importance
- Error breakdown by segment


## Machine Learning Models

The following models were implemented:

- Random Forest Regressor
- Linear Regression 
- Naive baseline

The naive baseline serves as a baseline to compare against the Random Forest model and Linear Regression.


## Features Used

Some features used include:

- Current utilization rate
- Traffic congestion index
- Hour of day
- Peak hour indicator
- Charger type
- Charging network
- Location type
- Nearby amenities

Leakage features such as occupied ports, pricing, and estimated wait times were removed prior to training.


## Evaluation Metrics

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Additional validation includes:

- Residual analysis
- Feature importance
- Baseline comparison
- Overfitting check using training and testing scores
- Error breakdown by segment to find biases 


## Future Improvements

Potential future work includes:

- XGBoost
- LightGBM
- LSTM time-series forecasting
- Hyperparameter optimization
- Cross-validation using TimeSeriesSplit
- Live EV charging station data integration


## Authors

Team Members:

- Charvisree Koripella
- Ananya Pal
