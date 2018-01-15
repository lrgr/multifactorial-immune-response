#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, argparse, logging, pandas as pd, numpy as np, json
from sklearn.model_selection import *

# Load our modules
from models import EN, RF, MODEL_NAMES, PIPELINES, PARAM_GRIDS
from i_o import getLogger

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--feature_file', type=str, required=True)
parser.add_argument('-fcf', '--feature_class_file', type=str, required=True)
parser.add_argument('-of', '--outcome_file', type=str, required=True)
parser.add_argument('-op', '--output_prefix', type=str, required=True)
parser.add_argument('-m', '--model', type=str, required=True, choices=MODEL_NAMES)
parser.add_argument('-mi', '--max_iter', type=int, required=False,
    default=1000000,
    help='ElasticNet only. Default is parameter used for the paper submission.')
parser.add_argument('-t', '--tol', type=float, required=False,
    default=1e-7,
    help='ElasticNet only. Default is parameter used for the paper submission.')
parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)
parser.add_argument('-nj', '--n_jobs', type=int, default=1, required=False)
parser.add_argument('-tc', '--training_classes', type=str, required=False, nargs='*',
    default=['Clinical','Tumor','Blood'])
args = parser.parse_args(sys.argv[1:])

# Set up logger
logger = getLogger(args.verbosity)

# Load the input data
X = pd.read_csv(args.feature_file, index_col=0, sep='\t')
y = pd.read_csv(args.outcome_file, index_col=0, sep='\t')
feature_classes = pd.read_csv(args.feature_class_file, index_col=0, sep='\t')

# Align the features and outcomes
patients = X.index
X = X.reindex(index = patients)
y = y.reindex(index = patients)
outcome_name = y.columns[0]

# Create some data structures to hold our output
json_output = dict(patients=list(map(float, patients)), params=vars(args))

################################################################################
# TRAIN A MODEL ON ALL THE DATA
################################################################################
# Choose which feature classes to use in training;
# to use all feature classes set training_classes = ['Clinical','Tumor','Blood']
training_cols = feature_classes['Class'].isin(args.training_classes).index.tolist()

# Set up nested validation for parameter selection and eventual evaluation
# Define parameter selection protocol
pipeline = PIPELINES[args.model]
pipeline.named_steps['estimator'].set_params(n_jobs=args.n_jobs)
if args.model == EN:
    pipeline.named_steps['estimator'].set_params(max_iter=args.max_iter)
    pipeline.named_steps['estimator'].set_params(tol=args.tol)
param_grid = PARAM_GRIDS[args.model]
if param_grid is not None:
    # Perform parameter selection using inner loop of CV
    inner_cv = LeaveOneOut()
    gscv = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                        cv=inner_cv, n_jobs=args.n_jobs,
                        scoring = 'neg_mean_squared_error')
else:
    # No parameter selection required
    gscv = pipeline

# Produce held-out predictions for parameter-selected model
# using outer loop of CV
logger.info('* Performing LOO cross-validation...')
outer_cv = LeaveOneOut()
sys.stderr = sys.stdout
preds = pd.Series(cross_val_predict(estimator = gscv,
                                   X=X.loc[:,training_cols],
                                    y=y[outcome_name], cv=outer_cv,
                                    n_jobs = args.n_jobs,
                                    verbose=51 if args.verbosity > 0 else 0),
                 index = patients)

# Visualize and asses held-out predictions
# 1) Subset predictions and ground truth to relevant indices
sub_preds = preds.loc[patients].values
sub_y = y.loc[patients][outcome_name].values
min_val = min(sub_preds.min(), sub_y.min())
max_val = max(sub_preds.max(), sub_y.max())

# 2) Compare held-out RMSE to baseline RMSE obtained by predicting mean
baseline_sqd_err = (y[outcome_name] - y[outcome_name].mean())**2
pred_sqd_err = (sub_y-sub_preds)**2
logger.info('[Held-out RMSE, Baseline RMSE]: {}'.format([np.sqrt(pred_sqd_err.mean()),
                                                   np.sqrt(baseline_sqd_err.mean())]))
logger.info('[Held-out MSE, Baseline MSE]: {}'.format([pred_sqd_err.mean(), baseline_sqd_err.mean()]))
logger.info('[Held-out MAE, Baseline MAE]: {}'.format([np.sqrt(pred_sqd_err).mean(),
                                                 np.sqrt(baseline_sqd_err).mean()]))
variance_explained = 1. - pred_sqd_err.mean()/baseline_sqd_err.mean()
logger.info('Variance explained: {}'.format(variance_explained))

# 3) Record the data into our plots dictionary
json_output['ExpandedClones'] = {
    "preds": sub_preds.tolist(),
    "true": sub_y.tolist(),
    "variance_explained": variance_explained,
    "rmse": {
        "baseline": np.sqrt(baseline_sqd_err.mean()),
        "held-out": np.sqrt(pred_sqd_err.mean())
    },
    "mse": {
        "baseline": baseline_sqd_err.mean(),
        "held-out": pred_sqd_err.mean()
    },
    "mae": {
        "baseline": np.sqrt(baseline_sqd_err).mean(),
        "held-out": np.sqrt(pred_sqd_err).mean()
    }
}

################################################################################
# EVALUATE FEATURE IMPORTANCE
################################################################################
# Train each model on full dataset
logger.info('* Training model on all the data...')
pipeline.named_steps['estimator'].set_params(verbose=1 if args.verbosity else 0)
model = pipeline.fit(X.loc[:,training_cols], y[outcome_name])

# Examine variable importance or coefficients in each model.
# Weight raw variable coefficients by associated variable standard deviations;
# this places all variables on the same scale.
if args.model == RF:
    variable_scores = model.named_steps['estimator'].feature_importances_
elif args.model == EN:
    variable_scores = model.named_steps['estimator'].coef_ * X.loc[:,training_cols].fillna(X.loc[:,training_cols].median()).std()
else:
    raise NotImplementedError('Model "%s" not implemented.' % args.model)
variable_scores = pd.Series(variable_scores, index = X.loc[:, training_cols].columns, name='Score')

# Associate feature classes with scores
variable_scores = pd.concat([variable_scores, feature_classes], axis = 1)

# Sort scores by importance magnitude
variable_scores = variable_scores.reindex(variable_scores['Score'].abs().sort_values(ascending=False).index)

# Output a pretty summary of feature importances
var_importance_tbl = variable_scores.to_string()
rows = var_importance_tbl.split('\n')
logger.info('-' * len(rows[0]))
logger.info('RandomForest feature importances' if args.model == RF else 'ElasticNet coefficients')
logger.info('-' * len(rows[0]))
for row in rows:
    logger.info(row)
logger.info('')

################################################################################
# OUTPUT TO FILE
################################################################################
# Output plot data to JSON
with open(args.output_prefix + '-results.json', 'w') as OUT:
    json.dump( json_output, OUT )

# Output results summary to TSV
variable_scores.to_csv(args.output_prefix + '-coefficients.tsv', sep='\t', index=True)
