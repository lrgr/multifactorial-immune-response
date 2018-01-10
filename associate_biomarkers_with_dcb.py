#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, argparse, logging, pandas as pd, numpy as np, json
from i_o import getLogger

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--feature_file', type=str, required=True)
parser.add_argument('-rf', '--results_file', type=str, required=True)
parser.add_argument('-of', '--output_file', type=str, required=True)
parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)
args = parser.parse_args(sys.argv[1:])

# Set up logger
logger = getLogger(args.verbosity)

################################################################################
# EXAMINE THE ASSOCIATION BETWEEN PREDICTIONS AND DCB
################################################################################
# Load the input files
with open(args.results_file, 'r') as IN:
    results = json.load(IN)
    patients = results['patients']
    preds = results['ExpandedClones']['preds']

df = pd.read_csv(args.feature_file, sep='\t', index_col=0)

# Index and exponentiate predictions so they can be merged with the dataframe
exp_preds = pd.DataFrame(np.exp(pd.Series(preds, index=patients,
                        name='Predicted N Expanded Clones that were TILs A->B')))

# Add predictions to the dataframe
df = pd.merge(df, exp_preds, how='outer',left_index=True, right_index=True)

# Plot biomarker association with DCB for various biomarkers
biomarkers = ['N Expanded Clones that were TILs A->B',
              'Predicted N Expanded Clones that were TILs A->B',
              'expressed_neoantigen_count','missense_snv_count','PD-L1']
biomarker_plot_items = []
for biomarker in biomarkers:
    for x, y in zip(df['benefit'].tolist(), df[biomarker].tolist()):
        biomarker_plot_items.append({
            "Biomarker": biomarker,
            "Benefit": bool(x),
            "Biomarker value": y
        })

# Save the data to our plots dictionary
with open(args.output_file, 'w') as OUT:
    json.dump( dict(Biomarkers=biomarker_plot_items, params=vars(args)), OUT)
