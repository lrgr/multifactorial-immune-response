#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, argparse, logging, pandas as pd, numpy as np, json
from models import pipeline, estimator, param_grid
from sklearn.model_selection import *
from i_o import getLogger

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--feature_file', type=str, required=True)
parser.add_argument('-fcf', '--feature_class_file', type=str, required=True)
parser.add_argument('-of', '--outcome_file', type=str, required=True)
parser.add_argument('-op', '--output_file', type=str, required=True)
parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)
parser.add_argument('-rs', '--random_seed', type=int, default=12345, required=False)
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

# Restrict to the training columns
training_cols = feature_classes['Class'].isin(args.training_classes).index.tolist()

################################################################################
# RUN PERMUTATION TEST
################################################################################
# Conduct a permutation test of cross validation significance
if param_grid is not None:
    # Perform parameter selection using inner loop of CV
    inner_cv = LeaveOneOut()
    gscv = GridSearchCV(estimator=pipelines[key], param_grid=param_grids[key],
                            cv=inner_cv,
                            scoring = 'neg_mean_squared_error')
else:
    gscv = pipeline

# Convert dataframes to matrices to avoid dataframe splitting error
score, permutation_scores, pvalue = permutation_test_score(estimator = gscv,
    X=X.loc[:,training_cols].as_matrix(), y=y.as_matrix(), cv=outer_cv,
    n_permutations=n_permutations, n_jobs=args.n_jobs,
    random_state=args.random_seed, verbose=3,
    scoring = 'neg_mean_squared_error')

################################################################################
# OUTPUT TO FILE
################################################################################
with open(args.output_file, 'w') as OUT:
    output = dict(permutation_scores=permutation_scores, true_score=score,
                  n_permutations=n_permutations, pvalue=pvalue, params=vars(args))
    json.dump( output, OUT )
