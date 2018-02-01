#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd, numpy as np
from permutation_test import OUTCOME_TEST

# Model summary
def model_summary(args):
    assert( len(args.results_files) == len(args.excluded_feature_classes))

    # Load the results files to construct dataframe
    items = []
    for result_file, excluded_feature_class in zip(args.results_files, args.excluded_feature_classes):
        with open(result_file, 'r') as IN:
            obj = json.load(IN)
            mses = obj['mse']
            rmses = obj['rmse']
            var_explained = obj['variance_explained']
            maes = obj['mae']
            n_features = len(obj['training_features'])
            item = { "Excluded Feature Classes": excluded_feature_class.capitalize(), "No. features": n_features }
            for metric_name, metric in zip(['mae', 'mse', 'rmse'], [mses, rmses, maes]):
                item[metric_name.upper()] = metric['held-out']
            item['Variance explained'] = var_explained
            items.append(item)

    # Convert to DataFrame and output to file
    df = pd.DataFrame(items)
    df = df.sort_values('Variance explained', ascending=False)
    df = df[['Excluded Feature Classes', 'No. features', 'Variance explained', 'MSE', 'RMSE', 'MAE']]
    df.to_csv(args.output_file, sep='\t', index=False)

# Permutation summary
def permutation_summary(args):
    items = []
    for input_file in args.input_files:
        # Load the file
        with open(input_file, 'r') as IN:
            obj = json.load(IN)
            params = obj.get('params')

            #
            if params.get('test_type') == OUTCOME_TEST:
                feature_classes = ''
            else:
                feature_classes = ' '.join(map(str.capitalize, obj.get('feature_classes')))

        # Extract relevant information
        items.append({
            'Test type': params.get('test_type').capitalize(),
            'N': obj.get('n_permutations'),
            'Feature class': feature_classes,
            'Score': obj.get('true_score'),
            'P-value': obj.get('pvalue'),
            'Mean Permuted Score': np.mean(obj.get('permutation_scores'))
        })

    # Convert to DataFrame and save
    df = pd.DataFrame(items)[['Test type', 'Feature class', 'Score', 'N', 'Mean Permuted Score', 'P-value']]
    df = df.sort_values('Test type')
    df.to_csv(args.output_file, sep='\t', index=False)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-of', '--output_file', type=str, required=True)
    subparser = parser.add_subparsers(dest='mode', help='Model or permutation test.')

    model_parser = subparser.add_parser('model')
    model_parser.add_argument('-rf', '--results_files', nargs='*', type=str, required=True)
    model_parser.add_argument('-efc', '--excluded_feature_classes', type=str, nargs='*', required=True)

    permute_parser = subparser.add_parser('permutation')
    permute_parser.add_argument('-i', '--input_files', type=str, required=True, nargs='*')
    args = parser.parse_args(sys.argv[1:])

    # Call the appropriate summarizer
    if args.mode == 'model':
        model_summary(args)
    elif args.mode == 'permutation':
        permutation_summary(args)
    else:
        raise NotImplementedError('Mode "%s" not implemented.' % args.mode)
