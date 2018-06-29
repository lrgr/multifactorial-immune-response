#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-rf', '--results_files', nargs='*', type=str, required=True)
parser.add_argument('-of', '--output_file', type=str, required=True)
parser.add_argument('-efc', '--excluded_feature_classes', type=str, nargs='*', required=True)
args = parser.parse_args(sys.argv[1:])

assert( len(args.results_files) == len(args.excluded_feature_classes))

# Load the results files to construct dataframe
items = []
for result_file, excluded_feature_class in zip(args.results_files, args.excluded_feature_classes):
    with open(result_file, 'r') as IN:
        obj = json.load(IN)
        aurocs = obj['AUROC']
        auprs = obj['AUPR']
        accuracies = obj['accuracy']
        n_features = len(obj['training_features'])
        item = { "Excluded Feature Classes": excluded_feature_class.capitalize(), "No. features": n_features }
        for metric_name, metric in zip(['AUROC', 'AUPR', 'Accuracy'], [aurocs, auprs, accuracies]):
            item[metric_name] = metric
        items.append(item)

# Convert to DataFrame and output to file
df = pd.DataFrame(items)
df = df.sort_values('AUROC', ascending=False)
df = df[['Excluded Feature Classes', 'No. features', 'AUROC', 'AUPR', 'Accuracy']]
df.to_csv(args.output_file, sep='\t', index=False)
