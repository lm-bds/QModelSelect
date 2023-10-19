import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold

from preprocess import DataPreprocessor
from tune import TuneModel
from graph import Grapher
from models import create_models
from preprocess import ColumnSelector
from metrics import mean_squared_error, calculate_rmse

DATA_PATH = ".//data//data.csv"
COLUMNS_TO_DROP = [] #columns to remove
COLUMNS_TO_ENCODE = [] # column that require one-hot encoding
TARGET = '' # the target of the regression model
CORRELATION_THRESHOLD = 0.1 #drop columns that dont correlate with the target above this 
MODELS = ["Random Forest",
          "SVR",
          "Decision Tree",
          "Gradient Boosting",
          "Lasso Regression",
          "SVC",
          "Ridge Regression",
          "K-Nearest Neighbors",
          "MLP Regressor", 
          "Bayesian Ridge Regression",
          "Huber Regressor",
          "Passive Aggressive Regressor",
          "Theil-Sen Regressor",
          "Orthogonal Matching Pursuit",
          "Isotonic Regression"
          ]

# Preprocessing
data = pd.read_csv(DATA_PATH)
corr_grapher = Grapher()
preprocessor = DataPreprocessor(COLUMNS_TO_ENCODE, COLUMNS_TO_DROP)
cleaned_data = preprocessor.load_data("diabetic_data.csv").clean().get_data()
column_selector = ColumnSelector(cleaned_data,CORRELATION_THRESHOLD, TARGET)
selected_data = column_selector.selector()

selected_correlation_matrix = selected_data.corr()
corr_grapher.CorrMatrix.corr_matrix(selected_data)


features = selected_data.drop(columns=[TARGET])
target = selected_data[TARGET]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


selected_models = create_models(MODELS)


####

###Create Baseline for comparision
mean_target = np.mean(y_test)
median_target = np.median(y_test)

y_pred_mean = np.full_like(y_test, mean_target)
rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))

print(f"The Naive Baseline RMSE using mean prediction is: {rmse_mean}")

y_pred_median = np.full_like(y_test, median_target)
rmse_median = np.sqrt(mean_squared_error(y_test, y_pred_median))

print(f"The Naive Baseline RMSE using median prediction is: {rmse_median}")
####


for model_name, model_pipeline, tuning_grid in selected_models:
    print(f"Testing {model_name}")
    model = TuneModel(model_pipeline, X_train, y_train, X_test, y_test, tuning_grid, cv)

    tuned_model = model.fit_model()




    model_attributes = tuned_model.get_params()

    for attribute, value in model_attributes.items():
        print(f"{attribute}: {value}")


    predictions = tuned_model.predict(X_test)
    rmse = calculate_rmse(tuned_model, X_test, y_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"RMSE for the tuned {model_name} model is {rmse:.2f}\n")
    print(f"The Mean Squared Error for the tuned {model_name} model is {mse:.2f}\n")




    prediction_plot = Grapher.plot_predictions(tuned_model, X_test, y_test,"Predicted vs real values")