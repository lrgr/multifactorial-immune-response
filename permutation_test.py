#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, argparse, logging, pandas as pd, numpy as np, json
from models import MODEL_NAMES, init_model
from i_o import getLogger

################################################################################
# MAP
################################################################################
# Run the permutation test
def map_permutation_test(args):
    # Set up logger
    logger = getLogger(args.verbosity)

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

    # Restrict to the training columns
    training_cols = feature_classes['Class'].isin(args.training_classes).index.tolist()

    ############################################################################
    # RUN PERMUTATION TEST
    ############################################################################
    #Initialize model
    pipeline, gscv = init_model(args.model, args.n_jobs, args.estimator_random_seed,
        args.max_iter, args.tol)

    # Permute the outcomes
    np.random.seed(args.permutation_random_seed)
    y[outcome_name] = np.random.permutation(y[outcome_name])

    # Convert dataframes to matrices to avoid dataframe splitting error
    outer_cv = LeaveOneOut()
    preds = pd.Series(cross_val_predict(estimator = gscv,
                                       X=X.loc[:,training_cols],
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
            "var_explained": var_explained.tolist(),
            "true": sub_y.tolist(),
            "preds": "sub_preds",
            "params": vars(args)
        }
        output.update(metric_vals.items())
        json.dump( output, OUT )

################################################################################
# REDUCE
################################################################################
#
def reduce_permutation_test(args):
    ############################################################################
    # LOAD AND SUMMARIZE INPUT
    ############################################################################
    # Set up logger
    logger = getLogger(args.verbosity)

    # Load results file
    with open(args.results_file, 'r') as IN:
        print(args.results_file)
        obj = json.load(IN)
        true_score = obj['mse']['held-out']

    # Load permuted results files
    permutation_scores = []
    for permuted_results_file in args.permuted_results_files:
        with open(permuted_results_file, 'r') as IN:
            permutation_scores.append( json.load(IN)['mse']['held-out'] )

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
                      pvalue=pvalue, params=vars(args))
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
    map_parser.add_argument('-tc', '--training_classes', type=str, required=False, nargs='*',
        default=['Clinical','Tumor','Blood'])

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
        raise NotImplementedError('Mode "%s" not implemented.' % args.mode)

if __name__ == '__main__':
    run( get_parser().parse_args(sys.argv[1:]) )
