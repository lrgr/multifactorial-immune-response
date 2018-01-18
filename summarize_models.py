#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-rf', '--results_files', nargs='*', type=str, required=True)
parser.add_argument('-of', '--output_file', type=str, required=True)
parser.add_argument('-sfc', '--selected_feature_classes', type=str, nargs='*', required=True)
args = parser.parse_args(sys.argv[1:])

assert( len(args.results_files) == len(args.selected_feature_classes))

# Load the results files to construct dataframe
items = []
for result_file, feature_class in zip(args.results_files, args.selected_feature_classes):
    with open(result_file, 'r') as IN:
        obj = json.load(IN)
        mses = obj['mse']
        rmses = obj['rmse']
        var_explained = obj['variance_explained']
        maes = obj['mae']
        n_features = len(obj['training_features'])
        item = { "Excluded Feature Classes": feature_class.capitalize(), "No. features": n_features }
        for metric_name, metric in zip(['mae', 'mse', 'rmse'], [mses, rmses, maes]):
            item[metric_name.upper()] = metric['held-out']
        item['Variance explained'] = var_explained
        items.append(item)

# Convert to DataFrame and output to file
df = pd.DataFrame(items)
df = df.sort_values('Variance explained', ascending=False)
df = df[['Excluded Feature Classes', 'No. features', 'Variance explained', 'MSE', 'RMSE', 'MAE']]
df.to_csv(args.output_file, sep='\t', index=False)
