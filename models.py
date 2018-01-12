#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold

# Constants
EN = 'en'
RF = 'rf'
MODEL_NAMES = [ EN, RF ]
MODEL_TO_NAME = {
    EN: 'ElasticNet',
    RF: 'RandomForest'
}
IMPORTANCE_NAMES = {
    EN: 'Learned coefficient',
    RF: 'Variable importance'
}

################################################################################
# ELASTIC NET
################################################################################
# Elastic net with LOOCV
en_inner_cv = LeaveOneOut()
en_estimator = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99],
                        cv = en_inner_cv, normalize=True,
                        max_iter = 100000, tol=1e-7)
en_param_grid = None

# Build pipeline which performs estimation after imputing missing values
en_pipeline = Pipeline([("imputer", Imputer(strategy="median")),
                     ("estimator", en_estimator)])

################################################################################
# RANDOM FOREST
################################################################################
# Elastic net with LOOCV
rf_inner_cv = LeaveOneOut()
rf_seed = 12345
rf_estimator = RandomForestRegressor(n_estimators=1000, random_state=rf_seed)
rf_param_grid = None
rf_pipeline = Pipeline([("imputer", Imputer(strategy="median")),
                          ("estimator", rf_estimator)])

################################################################################
# PIPELINES
################################################################################
PIPELINES = {
    EN: en_pipeline,
    RF: rf_pipeline
}

PARAM_GRIDS = {
    EN: en_param_grid,
    RF: rf_param_grid
}
