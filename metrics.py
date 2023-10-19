import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def calculate_rmse(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse