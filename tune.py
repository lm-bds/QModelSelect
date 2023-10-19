from dataclasses import dataclass
import pandas as pd
import sklearn.pipeline

from sklearn.model_selection import GridSearchCV 

@dataclass
class ModelCharacteristics:
    models: list

class TuneModel:
    def __init__(self, pipeline : sklearn.pipeline.Pipeline, X_train: pd.DataFrame,
                 y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                 tuning_grid: dict,cv):
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.tuning_grid = tuning_grid
        self.cv = cv


    def fit_model(self):
        grid_search = GridSearchCV(self.pipeline, self.tuning_grid, cv=self.cv, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        # Fit the best estimator
        best_estimator.fit(self.X_train, self.y_train)
        
        y_pred = best_estimator.predict(self.X_test)

        print("Best Parameters:", best_params)


        return best_estimator
