# A Multifactorial Model of T Cell Expansion and Durable Clinical Benefit in Response to a PD-L1 Inhibitor
<img src='https://travis-ci.org/lrgr/multifactorial-immune-response.svg?branch=master'>

This repository contains the source code for reproducing the experiments and figures from Leiserson, et al. (bioRxiv, 2017). See the [references section](https://github.com/lrgr/multifactorial-immune-response#references) below for more information on the paper.

### Setup

#### Dependencies
The methods and experiments are written in Python 3. We recommend using Conda to manage dependencies, which you can do directly using the provided `environment.yml` file:

    conda env create -f environment.yml
    source activate hierarchical-hmm-env

### Usage

We use [`snakemake`](https://snakemake.readthedocs.io/en/latest/) to run a pipeline of commands to download and process data, run experiments, and create the figures. To run the entire pipeline, simply run:

    snakemake all

### Support

Please report bugs and feature requests in the [Issues tab](https://github.com/lrgr/multifactorial-immune-response/issues) of this GitHub repository.

For further questions, please email [Max Leiserson](mailto:mdml@cs.umd.edu) and [Lester Mackey](mailto:lmackey@microsoft.com) directly.

### References

Mark DM Leiserson, Vasilis Syrgkanis, Amy Gilson, Miroslav Dudik, Samuel Funt, Alexandra Snyder, Lester Mackey. (2018) A Multifactorial Model of T Cell Expansion and Durable Clinical Benefit in Response to a PD-L1 Inhibitor. _In submission_ [[bioRxiv preprint]](https://www.biorxiv.org/content/early/2017/12/08/231316).
