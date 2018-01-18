from os.path import join
configfile: 'configs/default.yml'

################################################################################
# SETTINGS, FILES, AND DIRECTORIES
################################################################################
# Directories
DATA_DIR = 'data'
RAW_DATA_DIR = join(DATA_DIR, 'raw')
OUTPUT_DIR = 'output/%s' % config['model']
FIGURES_DIR = join(OUTPUT_DIR, 'figs')
PERMUTATION_DIR = join(OUTPUT_DIR, 'permutations')
MODELS_DIR = join(OUTPUT_DIR, 'models')

# Data files
SNYDER_COUNTS   = join(RAW_DATA_DIR, 'snyder_et_al_data_counts.csv')
SNYDER_CLINICAL = join(RAW_DATA_DIR, 'snyder_et_al_data_clinical.csv')
SNYDER_TCR      = join(RAW_DATA_DIR, 'snyder_et_al_data_tcr.csv')
SNYDER_TCR_EXPANSION_AB = join(RAW_DATA_DIR, 'snyder_et_al_data_tcr_expansion_a_b.csv')
SNYDER_TCR_EXPANSION_AC = join(RAW_DATA_DIR, 'snyder_et_al_data_tcr_expansion_a_c.csv')
SNYDER_FEATURE_CLASSES = join(DATA_DIR, 'snyder_et_al_feature_classes.xlsx')

PROCESSED_DATA_PREFIX = join(DATA_DIR, 'snyder_et_al_processed_data')
PROCESSED_FEATURES = PROCESSED_DATA_PREFIX + '-features.tsv'
PROCESSED_ALL_FEATURES = PROCESSED_DATA_PREFIX + '-all-features.tsv'
PROCESSED_OUTCOMES = PROCESSED_DATA_PREFIX + '-outcomes.tsv'
PROCESSED_FEATURE_CLASSES = PROCESSED_DATA_PREFIX + '-feature-classes.tsv'

# Output files
MODEL_OUTPUT_PREFIX = join(MODELS_DIR, '%s-trained' % config['model'])
MODEL_RESULTS = MODEL_OUTPUT_PREFIX + '-results.json'
MODEL_COEFFICIENTS = MODEL_OUTPUT_PREFIX + '-coefficients.tsv'

BIOMARKER_DCB_PLOT_OUTPUT = join(OUTPUT_DIR, '%s-biomarker-dcb-plot-data.json' % config['model'])

PERMUTATION_TEST_PREFIX = join(PERMUTATION_DIR, '%s-permuted-results' % config['model'])
PERMUTATION_TEST_RESULTS = join(OUTPUT_DIR, '%s-permutation-test-results.json' % config['model'])

# Plots
FIGS_PREFIX = join(FIGURES_DIR, 'fig')
FIG1 = '%s1.%s' % (FIGS_PREFIX, config['figure_format'])
FIG2 = '%s2.%s' % (FIGS_PREFIX, config['figure_format'])
FIG3 = '%s3.%s' % (FIGS_PREFIX, config['figure_format'])

################################################################################
# RULES
################################################################################
# Data processing and download
rule download_data:
    params:
        url='https://raw.githubusercontent.com/hammerlab/multi-omic-urothelial-anti-pdl1/master/data_{datatype}.csv'
    output:
        join(RAW_DATA_DIR, 'snyder_et_al_data_{datatype}.csv')
    shell:
        'wget -O {output} {params.url}'

rule process_data:
    input:
        counts=SNYDER_COUNTS,
        clinical=SNYDER_CLINICAL,
        tcr=SNYDER_TCR,
        expansion_ab=SNYDER_TCR_EXPANSION_AB,
        expansion_ac=SNYDER_TCR_EXPANSION_AC,
        feature_classes=SNYDER_FEATURE_CLASSES
    output:
        PROCESSED_FEATURES,
        PROCESSED_OUTCOMES,
        PROCESSED_FEATURE_CLASSES,
        PROCESSED_ALL_FEATURES
    shell:
        'python construct_dataset.py -cof {input.counts} -clf {input.clinical} '\
        '-fcf {input.feature_classes} -tf {input.tcr} -ef {input.expansion_ab} '\
        '{input.expansion_ac} -op {PROCESSED_DATA_PREFIX}'

# Train model
rule train_model:
    input:
        features=PROCESSED_FEATURES,
        outcomes=PROCESSED_OUTCOMES,
        feature_classes=PROCESSED_FEATURE_CLASSES
    params:
        model=config['model'],
        n_jobs=config['n_jobs'],
        max_iter=config['max_iter'],
        tol=config['tol'],
        random_seed=config['random_seed']
    threads: config['n_jobs']
    output:
        MODEL_COEFFICIENTS,
        MODEL_RESULTS
    shell:
        'python train_model.py -ff {input.features} -fcf {input.feature_classes} '\
        '-of {input.outcomes} -op {MODEL_OUTPUT_PREFIX} -m {params.model} '\
        '-nj {params.n_jobs} -mi {params.max_iter} -t {params.tol} '\
        '-rs {params.random_seed}'

# Do follow up analysis
rule biomarkers_and_dcb:
    input:
        features=PROCESSED_ALL_FEATURES,
        results=MODEL_RESULTS
    output:
        BIOMARKER_DCB_PLOT_OUTPUT
    shell:
        'python associate_biomarkers_with_dcb.py -ff {input.features} -rf {input.results} -o {output}'

rule map_permutation_test:
    input:
        features=PROCESSED_FEATURES,
        outcomes=PROCESSED_OUTCOMES,
        feature_classes=PROCESSED_FEATURE_CLASSES
    params:
        random_seed=lambda wildcards, output: config['random_seed'] + int(wildcards['index']),
        n_jobs=config['n_jobs'],
        model=config['model'],
        max_iter=config['max_iter'],
        tol=config['tol']
    threads: config['n_jobs']
    output:
        "%s-{index}.json" % PERMUTATION_TEST_PREFIX
    shell:
        'python permutation_test.py -of {output} map -ff {input.features} '\
        '-fcf {input.feature_classes} -ocf {input.outcomes}  '\
        '-nj {params.n_jobs} -ers {params.random_seed} -prs {params.random_seed}'\
        ' -m {params.model} -mi {params.max_iter} -t {params.tol}'

rule reduce_permutation_test:
    input:
        permutation_test_files=expand("%s-{index}.json" % PERMUTATION_TEST_PREFIX, index=range(1, config['n_permutations']+1)),
        results_file=MODEL_RESULTS
    output:
        PERMUTATION_TEST_RESULTS
    shell:
        'python permutation_test.py -of {output} reduce '\
        '-rf {input.results_file} -pf {input.permutation_test_files}'

# Make plots
rule plot:
    input:
        biomarkers=BIOMARKER_DCB_PLOT_OUTPUT,
        results=MODEL_RESULTS,
        permutation_test_results=PERMUTATION_TEST_RESULTS,
        coefficients=MODEL_COEFFICIENTS
    params:
        ext=config['figure_format']
    output:
        FIG1,
        FIG2,
        FIG3
    shell:
        'python plot_figures.py -bf {input.biomarkers} -rf {input.results} '\
        '-prf {input.permutation_test_results} -cf {input.coefficients} '\
        '-o {FIGS_PREFIX} -e {params.ext}'

# General
rule all:
    input:
        FIG1,
        FIG2,
        FIG3
