#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, argparse, logging, pandas as pd, numpy as np, json
from models import MODEL_NAMES, init_model, FEATURE_CLASSES
from i_o import getLogger

# Constants
FEATURE_TEST = 'feature'
OUTCOME_TEST = 'outcome'

################################################################################
# FEATURE PERMUTATION TEST
################################################################################
def shuffle_with_feat_mask(X, feat_mask, random_state):
    """Auxiliary function for feature_permutation_test"""
    indices = random_state.permutation(X.shape[0])
    X_perm = np.copy(X)
    X_feats = X[:, feat_mask]
    X_perm[:, feat_mask] = X_feats[indices, :]
    return X_perm

def feature_permutation_test(estimator, X, y, feat_mask, random_state=0,
                             cv=None, scoring=None, verbosity=logging.INFO):
    # Create logger if missing
    if logger is None:
        logger = getLogger(verbosity)

    # Sanity checking
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    # Shuffle data
    X_perm = shuffle_with_feat_mask(X, feat_mask, random_state)

    # Fit model on permuted data
    logger.info("Scoring permutation.")
    avg_score = []
    for train, test in cv.split(X, y):
        X_train, y_train = _safe_split(estimator, X_perm, y, train)
        X_test, y_test = _safe_split(estimator, X_perm, y, test, train)
        estimator.fit(X_train, y_train)
        avg_score.append(scorer(estimator, X_test, y_test))

    perm_score = np.mean(avg_score)
    logger.info("Permutation score: {}".format(perm_score))
    return perm_score

def map_feature_permutation_test(args):
    # Set up logger
    logger = getLogger(args.verbosity)

    # Additional argument checking
    if len(args.feature_classes) == 0:
        logger.error('Must provide a feature class for feature permutation test')
        assert False

    # Load required modules
    from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
    from metrics import compute_metrics

    # Load the input data
    X = pd.read_csv(args.feature_file, index_col=0, sep='\t')
    y = pd.read_csv(args.outcome_file, index_col=0, sep='\t')
    feature_classes = pd.read_csv(args.feature_class_file, index_col=0, sep='\t')

    # Align the features and outcomes
    patients = X.index
    X = X.reindex(index = patients)
    y = y.reindex(index = patients)
    outcome_name = y.columns[0]

    # In this case, we use all columns for training (just shuffle a subset)
    training_cols = feature_classes['Class'].tolist()
    feat_mask = feature_classes[training_cols].isin(args.feat_groups).as_matrix()

    ############################################################################
    # RUN PERMUTATION TEST
    ############################################################################
    #Initialize model
    pipeline, gscv = init_model(args.model, args.n_jobs, args.estimator_random_seed, args.max_iter, args.tol)

    # Convert dataframes to matrices to avoid dataframe splitting error
    logger.info("Running permutation test on features: {}".format(training_cols[feat_mask]))
    perm_score = feature_permutation_test(gscv, X.loc[:,training_cols].as_matrix(),
        y.as_matrix(), feat_mask, cv=outer_cv, n_permutations=n_permutations,
        n_jobs=args.n_jobs, random_state=perm_seed, verbose=3,
        scoring = 'neg_mean_squared_error')

    return perm_score

def reduce_feature_permutation_test(args):
    return

################################################################################
# OUTCOME PERMUTATION TEST
################################################################################
# Generate permuted data and train a model
def map_permutation_test(args):
    # Set up logger
    logger = getLogger(args.verbosity)

    # Load required modules
    from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
    from metrics import compute_metrics
    from sklearn.utils import check_random_state, safe_indexing

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
    logger.info('* Permuting %ss...' % args.test_type)
    np.random.seed(args.permutation_random_seed)
    if args.test_type == FEATURE_TEST:
        training_cols = feature_classes['Class'].index.tolist()
        permuted_features = feature_classes.loc[feature_classes['Class'].isin(map(str.capitalize, args.feature_classes))].index.tolist()
        for f in permuted_features:
            X[f] = np.random.permutation(X[f])

    # Permute the outcomes
    elif args.test_type == OUTCOME_TEST:
        feature_class_names = set(map(str.capitalize, set(FEATURE_CLASSES) - set(args.feature_classes)))
        training_cols = feature_classes.loc[feature_classes['Class'].isin(feature_class_names)].index.tolist()
        y[outcome_name] = np.random.permutation(y[outcome_name])
    else:
        raise NotImplementedError('Test type "%s" not implemented.' % args.test_type)

    ############################################################################
    # RUN PERMUTATION TEST
    ############################################################################
    #Initialize model
    pipeline, gscv = init_model(args.model, args.n_jobs,
        args.estimator_random_seed, args.max_iter, args.tol)

    # Convert dataframes to matrices to avoid dataframe splitting error
    outer_cv = LeaveOneOut()
    preds = pd.Series(cross_val_predict(estimator = gscv, X=X.loc[:,training_cols],
                                        y=y[outcome_name], cv=outer_cv,
                                        n_jobs = args.n_jobs,
                                        verbose=61 if args.verbosity > 0 else 0),
                     index = patients)

    # Evalue results
    sub_y = y.loc[patients][outcome_name].values
    sub_preds = preds.loc[patients].values
    metric_vals, var_explained = compute_metrics(sub_y, sub_preds)

    ############################################################################
    # OUTPUT TO FILE
    ############################################################################
    with open(args.output_file, 'w') as OUT:
        output = {
            "params": vars(args),
            "var_explained": var_explained.tolist(),
            "true": sub_y.tolist(),
            "preds": "sub_preds",
            "params": vars(args),
            "training_features": training_cols
        }
        output.update(metric_vals.items())
        json.dump( output, OUT )

# Read in a bunch of results on permuted data and compute significance
def reduce_permutation_test(args):
    ############################################################################
    # LOAD AND SUMMARIZE INPUT
    ############################################################################
    # Set up logger
    logger = getLogger(args.verbosity)

    # Load results file
    with open(args.results_file, 'r') as IN:
        obj = json.load(IN)
        true_score = obj['mse']['held-out']

    # Load permuted results files
    permutation_scores = []
    feature_classes = None
    for i, permuted_results_file in enumerate(args.permuted_results_files):
        with open(permuted_results_file, 'r') as IN:
            obj = json.load(IN)
            permutation_scores.append( obj['mse']['held-out'] )
            if i == 0 and args.test_type == FEATURE_TEST:
                feature_classes = obj.get('params').get('feature_classes')

    n_permutations = len(permutation_scores)

    # Compute P-value
    pvalue = (1. + sum(1. for s in permutation_scores if s >= true_score))/(n_permutations + 1.)
    logger.info('- No. permutations: %s' % n_permutations)
    logger.info('- True score: %.5f' % true_score)
    logger.info('- P-value: p < %s' % pvalue)

    ############################################################################
    # OUTPUT TO FILE
    ############################################################################
    with open(args.output_file, 'w') as OUT:
        output = dict(permutation_scores=permutation_scores,
                      true_score=true_score, n_permutations=n_permutations,
                      pvalue=pvalue, params=vars(args), feature_classes=feature_classes)
        json.dump( output, OUT )

################################################################################
# MAIN
################################################################################
# Command-line argument parser
def get_parser():
    # Set up and global arguments
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='mode', help='Map or reduce.')

    parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)
    parser.add_argument('-of', '--output_file', type=str, required=True)
    parser.add_argument('-tt', '--test_type', required=True,
        choices=[FEATURE_TEST, OUTCOME_TEST],
        help='Type of permutation test (feature or outcome).')

    # Mapping arguments
    map_parser = subparser.add_parser("map")
    map_parser.add_argument('-ff', '--feature_file', type=str, required=True)
    map_parser.add_argument('-fcf', '--feature_class_file', type=str, required=True)
    map_parser.add_argument('-ocf', '--outcome_file', type=str, required=True)
    map_parser.add_argument('-m', '--model', type=str, required=True, choices=MODEL_NAMES)
    map_parser.add_argument('-mi', '--max_iter', type=int, required=False,
        default=1000000,
        help='ElasticNet only. Default is parameter used for the paper submission.')
    map_parser.add_argument('-t', '--tol', type=float, required=False,
        default=1e-7,
        help='Default is parameter used for the paper submission.')
    map_parser.add_argument('-nj', '--n_jobs', type=int, default=1, required=False)
    map_parser.add_argument('-ers', '--estimator_random_seed', type=int,
        default=12345, required=False)
    map_parser.add_argument('-prs', '--permutation_random_seed', type=int,
        default=12345, required=False)
    map_parser.add_argument('-fc', '--feature_classes', type=str, required=False,
        default=[], choices=FEATURE_CLASSES, nargs='*',
        help='For outcome permutation test, these are feature classes to be excluded. '\
             'For feature permutation test, these are the features to permute.')

    # Reduce arguments
    reduce_parser = subparser.add_parser("reduce")
    reduce_parser.add_argument('-rf', '--results_file', type=str, required=True)
    reduce_parser.add_argument('-pf', '--permuted_results_files', type=str,
        nargs='*', required=False, default=[])

    return parser

def run(args):
    if args.mode == 'map':
        map_permutation_test(args)
    elif args.mode == 'reduce':
        reduce_permutation_test(args)
    else:
        raise NotImplementedError('Mode "%s" not implemented for permutation test.' % args.mode)

if __name__ == '__main__':
    run( get_parser().parse_args(sys.argv[1:]) )
