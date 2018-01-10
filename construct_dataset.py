#!/usr/bin/env python

################################################################################
# SET UP
################################################################################
# Load required modules
import sys, os, argparse, logging, pandas as pd, numpy as np
from i_o import getLogger

# Set up command-line parser.
# 1) Input files
parser = argparse.ArgumentParser()
parser.add_argument('-cof', '--counts_file', type=str, required=True)
parser.add_argument('-clf', '--clinical_file', type=str, required=True)
parser.add_argument('-tf', '--tcr_file', type=str, required=True)
parser.add_argument('-ef', '--expansion_files', type=str, required=True, nargs=2)
parser.add_argument('-fcf', '--feature_class_file', type=str, required=True)

# 2) Feature/Outcome parameters
parser.add_argument('-on', '--outcome_name', type=str, required=False,
    default='N Expanded Clones that were TILs A->B')
parser.add_argument('-fn', '--feature_names', type=str, required=False,
    nargs='*', default=['Age', 'Albumin < 4',
            'Time since last chemotherapy','missense_snv_count',
            'expressed_missense_snv_count', 'neoantigen_count',
            'expressed_neoantigen_count','T-cell fraction',
            'Clonality', 'Diversity', 'Productive Unique TCRs (cnt)',
            'T-cell fraction_tumor', 'Clonality_tumor', 'Diversity_tumor',
            'Top Clone Freq(%)', 'Number of chemo regimens total',
            'Baseline neutrophil to lymphocyte ratio', 'Prior BCG'])
parser.add_argument('-bfn', '--binary_feature_names', type=str, required=False,
    nargs='*', default=['Albumin < 4','Visceral Mets','Hb < 10','Liver Mets',
    'Previous perioperative chemotherapy, with first progression ²12 months',
    'Prior BCG', 'Smoking','Variant Histology'])
parser.add_argument('-ynf', '--yes_no_features', type=str, required=False,
    nargs='*', default=['Prior BCG', 'Smoking','Variant Histology',
    'Previous perioperative chemotherapy, with first progression ²12 months'])

# Output parameters
parser.add_argument('-op', '--output_prefix', type=str, required=True)
parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)

# Parse command-line arguments and set up logger
args = parser.parse_args(sys.argv[1:])
logger = getLogger(args.verbosity)

################################################################################
# LOAD AND MERGE THE INPUT FILES
################################################################################
# Load the counts and clinical data
counts   = pd.read_csv(args.counts_file)
clinical = pd.read_csv(args.clinical_file)
clinical_patient_ids = set(clinical['patient_id'])

# Load TCRs, restricting to time point A peripheral samples
tcrs = pd.read_csv(args.tcr_file)
tcrs = tcrs.loc[(tcrs['Sample Type'] == 'PBMC') & (tcrs['Time Point'] == 'A') & (tcrs['patient_id'].isin(clinical_patient_ids))]

tumor_tcrs = pd.read_csv(args.tcr_file)
tumor_tcrs = tumor_tcrs.loc[(tumor_tcrs['Sample Type'] == 'Tumor') & (tumor_tcrs['Time Point'] == 'A') & (tumor_tcrs['patient_id'].isin(clinical_patient_ids))]

# Merge the dataframes together
df = pd.merge(clinical, pd.merge(counts, pd.merge(tcrs,tumor_tcrs, suffixes=['','_tumor'], how='outer', on='patient_id'), how='outer', on='patient_id'), how='outer', on='patient_id')

# Load the measurements of TCRs in the blood at time points A and B
T_CELL_OUTCOMES = ['N Expanded Clones that were TILs', 'N Expanded Clones', 'Unique TIL clones in B(cnt)']
def load_t_cell_outcomes(time_point, expansion_file):
    # We want to rename the columns so we we know which time point we're
    # looking at
    col_names = { n: n + ' A->%s' % time_point for n in T_CELL_OUTCOMES[:-1] }

    # They always call it "Unique TIL clones in B(cnt)", no matter what time point,
    # and I think that is a bug so I automatically change it where appropriate
    if time_point != 'B':
        col_names['Unique TIL clones in B(cnt)'] = 'Unique TIL clones in %s(cnt)' % time_point

    # Load the table, restrict to the right patients, and rename the columns
    tcrs_blood_df = pd.read_csv(expansion_file)[['patient_id'] + T_CELL_OUTCOMES]
    tcrs_blood_df = tcrs_blood_df.loc[tcrs_blood_df['patient_id'].isin(clinical_patient_ids)]
    return tcrs_blood_df.rename(columns=col_names)

tcrs_blood_b = load_t_cell_outcomes('B', args.expansion_files[0])
tcrs_blood_c = load_t_cell_outcomes('C', args.expansion_files[1])

# Load the number of expanded clones that were originally in the tumor
# for each patient, and at each time point
df = pd.merge(pd.merge(df, tcrs_blood_b, how='outer', on='patient_id'), tcrs_blood_c, how='outer', on='patient_id')

# Print some summary statistics
logger.info('Number of patients')
logger.info('- Counts: %s %s' % (len(set(counts['patient_id'])), len(counts['patient_id'])))
logger.info('- Clinical: %s %s' % (len(set(clinical['patient_id'])), len(clinical['patient_id'])))
logger.info('- TCRs (tumor-only): %s %s' %  (len(set(tcrs['patient_id'])), len(tcrs['patient_id'])))
logger.info('- Merged: %s %s' % (len(set(df['patient_id'])), len(df['patient_id'])))
logger.info('- Columns: {}'.format(', '.join(df.columns.values)))

################################################################################
# SELECT OUR OUTCOME AND FEATURES
################################################################################
# Select outcome
y = df[['patient_id', args.outcome_name]].copy()
y = y.set_index('patient_id')

# Select features for prediction
X = df[args.feature_names + ['patient_id']].copy()
X = X.set_index('patient_id')

# Keep track of class of each feature
classes = pd.read_excel(args.feature_class_file, index_col='#Feature name')
feature_classes = classes.loc[args.feature_names].Class

# Employ one-hot encoding for categorical features
for feature in args.yes_no_features:
    if feature in X.columns:
        X[feature] = X[feature] == 'Y'

# Add maximum non-missing factor score as a feature
X['factor score'] = np.fmax(df['5-factor score'].values, df['2-factor score'].values)
feature_classes['factor score'] = classes.loc['5-factor score'].Class

# Add logged version of features
for feature in feature_classes.index:
    feature_classes['log_'+feature] = feature_classes[feature]
X = pd.concat([X, np.log(X.astype('float64')+1).add_prefix('log_')], axis = 1)

# Drop log versions of binary features, since they are redundant
for feature in args.binary_feature_names:
    if feature in X.columns:
        X.drop('log_'+feature, axis=1, inplace=True)
        feature_classes.drop('log_'+feature, inplace=True)

# Remove datapoints with missing outcomes
patients_missing_outcomes = y[y[args.outcome_name].isna()].index
X = X.drop(patients_missing_outcomes)
y = y.drop(patients_missing_outcomes)

# Place outcome on logarithmic scale
y = np.log(y)

# Print more summary statistics
logger.info('Final feature matrix')
logger.info('- %s patients (%s with missing outcomes removed )' % (X.shape[0], len(patients_missing_outcomes)))
logger.info('- %s features' % X.shape[1])

# Output data to file
df.to_csv(args.output_prefix + '-all-features.tsv', sep='\t', index=True)
X.to_csv(args.output_prefix + '-features.tsv', sep='\t', index=True)
y.to_csv(args.output_prefix + '-outcomes.tsv', sep='\t', index=True)
feature_classes = pd.DataFrame(feature_classes, columns=['Class'])
feature_classes.to_csv(args.output_prefix + '-feature-classes.tsv', sep='\t', index=True)
