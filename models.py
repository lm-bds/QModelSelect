from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import BayesianRidge, HuberRegressor, Lasso, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, Ridge, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor



@dataclass 
class ModelCharcteristics:
    name : str
    pipeline : Pipeline
    tuning_grid : dict


### RANDOM FOREST ###
rf_name = "Random Forest"

rf_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('rf', RandomForestRegressor(n_jobs=-1))
])

rf_tuning_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

## Model 2: SVR ##
svr_name = "SVR"

svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ("svr", SVR(kernel="poly", verbose=3))
])

svr_tuning_grid = {
    'svr__C': [0.1, 1, 5, 20],
    'svr__epsilon': [0.01, 0.1, 0.2]
}

## Model 3: Decision Tree ##
dt_name = "Decision Tree"

dt_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('dt', DecisionTreeRegressor())
])

dt_tuning_grid = {
    'dt__max_depth': [None, 10, 20, 30],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 2, 4]
}

## Model 4: Gradient Boosting ##
gb_name = "Gradient Boosting"

gb_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('gb', GradientBoostingRegressor())
])

gb_tuning_grid = {
    'gb__n_estimators': [100, 200, 300],
    'gb__max_depth': [3, 4, 5],
    'gb__min_samples_split': [2, 5, 10],
    'gb__min_samples_leaf': [1, 2, 4]
}

## Model 5: Lasso Regression ##
lasso_name = "Lasso Regression"

lasso_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('lasso', Lasso())
])

lasso_tuning_grid = {
    'lasso__alpha': [0.1, 0.5, 1.0, 2.0]
}

## Model 6: Support Vector Machine for Classification (SVC) ##
svc_name = "SVC"

svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ("svc", SVC(kernel="rbf"))
])

svc_tuning_grid = {
    'svc__C': [0.1, 1, 5, 10],
    'svc__gamma': ['scale', 'auto', 0.01, 0.1, 0.5]
}

## Model 7: Ridge Regression ##
ridge_name = "Ridge Regression"

ridge_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('ridge', Ridge())
])

ridge_tuning_grid = {
    'ridge__alpha': [0.1, 0.5, 1.0, 2.0]
}

## Model 8: K-Nearest Neighbors (KNN) ##
knn_name = "K-Nearest Neighbors"

knn_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('knn', KNeighborsRegressor())
])

knn_tuning_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance']
}

mlp_name = "MLP Regressor"

mlp_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('mlp', MLPRegressor())
])

mlp_tuning_grid = {
    'mlp__hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.0005, 0.001],
    'mlp__learning_rate': ['constant', 'invscaling', 'adaptive']
}

## Model 10: Bayesian Ridge Regression ##
bayesian_ridge_name = "Bayesian Ridge Regression"

bayesian_ridge_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('bayesian_ridge', BayesianRidge())
])

bayesian_ridge_tuning_grid = {
    'bayesian_ridge__n_iter': [300, 400, 500],
    'bayesian_ridge__alpha_1': [1e-6, 1e-5, 1e-4],
    'bayesian_ridge__alpha_2': [1e-6, 1e-5, 1e-4],
    'bayesian_ridge__lambda_1': [1e-6, 1e-5, 1e-4],
    'bayesian_ridge__lambda_2': [1e-6, 1e-5, 1e-4]
}

## Model 11: Huber Regressor ##
huber_name = "Huber Regressor"

huber_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('huber', HuberRegressor())
])

huber_tuning_grid = {
    'huber__epsilon': [1.1, 1.35, 1.5],
    'huber__alpha': [0.0001, 0.001, 0.01]
}

## Model 12: Passive Aggressive Regressor ##
passive_aggressive_name = "Passive Aggressive Regressor"

passive_aggressive_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('passive_aggressive', PassiveAggressiveRegressor())
])

passive_aggressive_tuning_grid = {
    'passive_aggressive__C': [0.1, 1, 5, 10],
    'passive_aggressive__epsilon': [0.01, 0.1, 0.2],
    'passive_aggressive__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
}

## Model 13: Theil-Sen Regressor ##
theil_sen_name = "Theil-Sen Regressor"

theil_sen_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('theil_sen', TheilSenRegressor())
])

theil_sen_tuning_grid = {
    'theil_sen__max_subpopulation': [100, 200, 300],
    'theil_sen__n_subsamples': [None, 10, 20, 30],
    'theil_sen__max_iter': [300, 500, 1000],
    'theil_sen__tol': [1e-4, 1e-3, 1e-2]
}

## Model 14: Orthogonal Matching Pursuit ##
orthogonal_matching_pursuit_name = "Orthogonal Matching Pursuit"

orthogonal_matching_pursuit_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('orthogonal_matching_pursuit', OrthogonalMatchingPursuit())
])

orthogonal_matching_pursuit_tuning_grid = {
    'orthogonal_matching_pursuit__n_nonzero_coefs': [10, 20, 30, 40],
    'orthogonal_matching_pursuit__tol': [1e-4, 1e-3, 1e-2]
}

## Model 15: Isotonic Regression ##
isotonic_name = "Isotonic Regression"

isotonic_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ("pca", PCA(n_components=0.90)),
    ('isotonic', IsotonicRegression())
])

isotonic_tuning_grid = {
    'isotonic__out_of_bounds': ['nan', 'clip'],
    'isotonic__increasing': [True, False]
}

models_params = ModelCharcteristics(
    (rf_name, rf_pipeline, rf_tuning_grid),
    (svr_name, svr_pipeline, svr_tuning_grid),
    (dt_name, dt_pipeline, dt_tuning_grid),
    (gb_name, gb_pipeline, gb_tuning_grid),
    (lasso_name, lasso_pipeline, lasso_tuning_grid),
    (svc_name, svc_pipeline, svc_tuning_grid),
    (ridge_name, ridge_pipeline, ridge_tuning_grid),
    (knn_name, knn_pipeline, knn_tuning_grid)
    (mlp_name, mlp_pipeline, mlp_tuning_grid),
    (bayesian_ridge_name, bayesian_ridge_pipeline, bayesian_ridge_tuning_grid),
    (huber_name, huber_pipeline, huber_tuning_grid),
    (passive_aggressive_name, passive_aggressive_pipeline, passive_aggressive_tuning_grid),
    (theil_sen_name, theil_sen_pipeline, theil_sen_tuning_grid),
    (orthogonal_matching_pursuit_name, orthogonal_matching_pursuit_pipeline, orthogonal_matching_pursuit_tuning_grid),
    (isotonic_name, isotonic_pipeline, isotonic_tuning_grid)
)

model_info = {
    "RANDOM FOREST": (rf_name, rf_pipeline, rf_tuning_grid),
    "SVR": (svr_name, svr_pipeline, svr_tuning_grid),
    "Decision Tree": (dt_name, dt_pipeline, dt_tuning_grid),
    "Gradient Boosting": (gb_name, gb_pipeline, gb_tuning_grid),
    "Lasso Regression": (lasso_name, lasso_pipeline, lasso_tuning_grid),
    "SVC": (svc_name, svc_pipeline, svc_tuning_grid),
    "Ridge Regression": (ridge_name, ridge_pipeline, ridge_tuning_grid),
    "K-Nearest Neighbors": (knn_name, knn_pipeline, knn_tuning_grid),
    "MLP Regressor": (mlp_name, mlp_pipeline, mlp_tuning_grid),
    "Bayesian Ridge Regression": (bayesian_ridge_name, bayesian_ridge_pipeline, bayesian_ridge_tuning_grid),
    "Huber Regressor": (huber_name, huber_pipeline, huber_tuning_grid),
    "Passive Aggressive Regressor": (passive_aggressive_name, passive_aggressive_pipeline, passive_aggressive_tuning_grid),
    "Theil-Sen Regressor": (theil_sen_name, theil_sen_pipeline, theil_sen_tuning_grid),
    "Orthogonal Matching Pursuit": (orthogonal_matching_pursuit_name, orthogonal_matching_pursuit_pipeline, orthogonal_matching_pursuit_tuning_grid),
    "Isotonic Regression": (isotonic_name, isotonic_pipeline, isotonic_tuning_grid),
}

def create_models(model_names):
    models = []
    for name in model_names:
        if name in model_info:
            models.append(ModelCharcteristics(*model_info[name]))
    return models

