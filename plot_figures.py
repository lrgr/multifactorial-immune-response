#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
import matplotlib.patches as mpatches
sns.set_style('whitegrid')

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('-o', '--output_prefix', type=str, required=True)
args = parser.parse_args(sys.argv[1:])

# Load the input file
with open(args.input_file, 'r') as IN:
    plot_data = json.load(IN)

###############################################################################
# FIGURE 1
###############################################################################
fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.set_size_inches(10, 5)

# Expanded clones (predicted vs. true)
pred = np.array(plot_data['ExpandedClones']['x'])
true = np.array(plot_data['ExpandedClones']['y'])
variance_explained = plot_data['ExpandedClones']['variance_explained']
ax1.scatter(pred, true)
min_val = min(pred.min(),true.min())
max_val = max(pred.max(),true.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'k-', color = 'r')
ax1.set_xlabel('Log predicted number of expanded clones', fontsize=16)
ax1.set_ylabel('Log held-out ground truth', fontsize=16)
ax1.text(0.01, 0.95, 'Variance explained: %.2f%%' % (variance_explained*100.),
        ha='left', va='top', transform=ax1.transAxes, fontsize=16)
ax1.set_title('(a)', fontsize=16)

# Permutation scores
permutation_scores = -1*np.array(plot_data['PermutationTest']['permutation_scores'])
true_score = -1*plot_data['PermutationTest']['true_score']
pvalue = plot_data['PermutationTest']['pvalue']

ax2.hist(permutation_scores, 20,
         label='Permuted (Monte \nCarlo $p < %.1g$)' % pvalue,
         edgecolor='black')
ylim = ax2.get_ylim()
ax2.plot(2 * [true_score], ylim, '--g', linewidth=3,
         label='True (%.3f)' % true_score)
ax2.set_ylim(ylim)
ax2.legend(fontsize=14)
ax2.set_xlabel('Leave-one-out mean squared error', fontsize=16)
ax2.set_title('(b)', fontsize=16)

# Save to file and clear
plt.tight_layout()
plt.savefig(args.output_prefix + '-fig1.pdf')
plt.savefig(args.output_prefix + '-fig1.png')
plt.clf()

###############################################################################
# FIGURE 2
###############################################################################
# Plot the variable importances (coloring by Class)
var_importance = pd.DataFrame(plot_data['VariableImportance'])
var_importance = var_importance.reset_index()
var_importance = var_importance.rename(index=str, columns={"index": "Feature", "score": "Learned coefficient"})
var_importance['Class'] = var_importance['Class'].map({'Blood': 'Circulating', 'Tumor': 'Tumor', 'Clinical': 'Clinical'})

#
classToColor = dict(zip(['Tumor', 'Circulating', 'Clinical'], sns.color_palette()[:3]))
featureToImportance = dict(zip(var_importance['Feature'], var_importance['Learned coefficient']))
featureToClass = dict(zip(var_importance['Feature'], var_importance['Class']))

features = sorted(var_importance['Feature'], key=lambda f: abs(featureToImportance[f]), reverse=True)
classes = [ featureToClass[f] for f in features ]
palette = [ classToColor[c] for c in classes ]
sns.set(font_scale=0.8, style='whitegrid')  # smaller
ax = sns.barplot(x="Learned coefficient", y="Feature", data=var_importance,
           label="Learned coefficient", palette=palette, order=features)
ax.set_xlabel(ax.get_xlabel(), fontsize=16)
ax.set_ylabel(ax.get_ylabel(), fontsize=16)

# Add custom legend
patches = [ mpatches.Patch(color=col, label=c) for c, col in classToColor.items() ]
plt.legend(handles=patches, fontsize=14)

# Output to file
plt.subplots_adjust(left=0.25, right=0.95, top=0.95)
plt.savefig(args.output_prefix + '-fig2.pdf')
plt.savefig(args.output_prefix + '-fig2.png')
plt.clf()
sns.set(font_scale=1, style='whitegrid')

###############################################################################
# FIGURE 3
###############################################################################
# Load the data and use nicer names
biomarker_nice_names = {
    "PD-L1": "PD-L1 expression",
    "missense_snv_count": "Missense SNV count",
    "expressed_neoantigen_count": "Expressed neoantigen count",
    "Predicted N Expanded Clones that were TILs A->B": "Predicted expanded TIL clones",
    "N Expanded Clones that were TILs A->B": "Expanded TIL clones"
}
biomarker_plot_items = plot_data['Biomarkers']
for item in biomarker_plot_items:
    item['Progression-free survival'] = '> 6 months' if item['Benefit'] else '≤ 6 months'
    item['Biomarker'] = biomarker_nice_names[item['Biomarker']]
    if type(item['Biomarker value']) == type('') and item['Biomarker value'].startswith('IC'):
        item['Biomarker value'] = int(item['Biomarker value'][2:])

biomarker_df = pd.DataFrame(biomarker_plot_items)
biomarker_df = biomarker_df.dropna(axis='rows', how='any')
biomarker_df = biomarker_df.loc[biomarker_df['Biomarker'] != 'Expanded TIL Clones']

# Plot with seaborn
ordered_biomarkers = ['Predicted expanded TIL clones', 'Missense SNV count', 'Expressed neoantigen count', 'PD-L1 expression']
g = sns.FacetGrid(biomarker_df, col="Biomarker", sharex=True, sharey=False,
                col_wrap=2, col_order=ordered_biomarkers)
g = g.map(sns.boxplot, "Progression-free survival", "Biomarker value",
    palette=sns.color_palette()[:2], width=0.4)
g = g.map(sns.swarmplot, "Progression-free survival", "Biomarker value", color="0.25")

# Custom y-axis for PD-L1 expression plot
g.axes[-1].set_yticks((0, 1, 2))
g.axes[-1].set_yticklabels(('<1%', '1-5%', '≥5%'))

# Prepend subfigure letter to titles
for biomarker, ax, letter in zip(ordered_biomarkers, g.axes, 'abcd'):
    ax.set_title('(%s)' % letter)
    ax.set_ylabel(biomarker)

# Show the plot
plt.tight_layout()
plt.savefig(args.output_prefix + '-fig3.pdf')
plt.savefig(args.output_prefix + '-fig3.png')
