
# A Multifactorial Model of T Cell Expansion and Durable Clinical Benefit in Response to a PD-L1 Inhibitor
<img src='https://travis-ci.org/lrgr/multifactorial-immune-response.svg?branch=master'>

This repository contains the source code for reproducing the experiments and figures from Leiserson, et al. (bioRxiv, 2017). See the [references section](#references) below for more information on the paper.

We use continuous integration to automatically regenerate the figures whenever there is a push to the master branch. You can see the [current figures below](#figures).

### Setup

#### Dependencies
The methods and experiments are written in Python 3. We recommend using Conda to manage dependencies, which you can do directly using the provided `environment.yml` file:

    conda env create -f environment.yml
    source activate hierarchical-hmm-env

### Usage

We use [`snakemake`](https://snakemake.readthedocs.io/en/latest/) to run a pipeline of commands to download and process data, run experiments, and create the figures. To run the entire pipeline, simply run:

    snakemake all

#### Configuration

Configuration for the entire pipeline is controlled by the variables in `config.yml`. The following variables can be set:

|Variable name      | Choices                | Default   |
|-------------------|------------------------|-----------|
| `model`           | `'en'`, `'rf'`         | `'en'`    |
| `n_permutations`  | Positive int           | 1         |
| `random_seed`     | Positive int           | 12345     |
| `n_jobs`          | Positive int           | 4         |
| `figure_format`   | Standard image formats | png       |


### Support

Please report bugs and feature requests in the [Issues tab](https://github.com/lrgr/multifactorial-immune-response/issues) of this GitHub repository.

For further questions, please email [Max Leiserson](mailto:mdml@cs.umd.edu) and [Lester Mackey](mailto:lmackey@microsoft.com) directly.

### References

Mark DM Leiserson, Vasilis Syrgkanis, Amy Gilson, Miroslav Dudik, Samuel Funt, Alexandra Snyder, Lester Mackey. (2018) A Multifactorial Model of T Cell Expansion and Durable Clinical Benefit in Response to a PD-L1 Inhibitor. _In submission_ [[bioRxiv preprint]](https://www.biorxiv.org/content/early/2017/12/08/231316).

### Figures

<img src='https://raw.githubusercontent.com/lrgr/multifactorial-immune-response/gh-pages/fig1.png' style='width:300px'>

**Figure 1**: (a) Predicted log TIL expansion versus ground-truth log TIL expansion for patients held out using LOOCV.  Predictions are formed using the elastic net. (b) Histogram of LOOCV error when patient responses are permuted uniformly at random 1000 times.  The overlaid dotted line displays the LOOCV error obtained on the original dataset.

 <img src='https://github.com/lrgr/multifactorial-immune-response/blob/gh-pages/fig2.png?raw=true' style='width:300px'>

**Figure 2**: Learned elastic net coefficients and feature types.

<img src='https://github.com/lrgr/multifactorial-immune-response/blob/gh-pages/fig3.png?raw=true' style='width:300px'>

**Figure 3**: Distributions of biomarker values in patients with and without durable clinical benefit (defined as ≥ 6 months of progression-free survival): (a) predicted number of expanded TIL clones; (b) missense SNV count; (c) expressed neoantigen count; and, (d) percentage of tumor infiltrating immune cells found to be PD-L1-positive.
