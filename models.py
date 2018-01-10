#!/usr/bin/env python

# Load required modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import LeaveOneOut, KFold

# Elastic net with LOOCV
inner_cv = LeaveOneOut()
estimator = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99],
                        cv = inner_cv, normalize=True,
                        max_iter = 100000)
param_grid = None

# Build pipeline which performs estimation after imputing missing values
pipeline = Pipeline([("imputer", Imputer(strategy="median")),
                     ("estimator", estimator)])
