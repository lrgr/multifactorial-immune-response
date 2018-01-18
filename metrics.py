#!/usr/bin/env python

# Load required modules
import sys, os, numpy as np

# Constants
RMSE = 'rmse'
MAE = 'mae'
MSE = 'mse'

# Metrics
def sqd_err(y, y_hat):
    return (y-y_hat)**2

def rmse(y, y_hat):
    return np.sqrt(sqd_err(y, y_hat).mean())

def mse(y, y_hat):
    return sqd_err(y, y_hat).mean()

def mae(y, y_hat):
    return np.sqrt(sqd_err(y, y_hat).mean())

def variance_explained(baseline_sqd_err, pred_sqd_err):
    return 1. - pred_sqd_err.mean()/baseline_sqd_err.mean()

metric_name_to_fn = {
    RMSE: rmse,
    MSE: mse,
    MAE: mae
}

# Helpers
def compute_metrics(y, y_hat, metric_names=[RMSE, MAE, MSE]):
    metric_vals = { metric_name: {} for metric_name in metric_names }
    y_mean = y.mean()
    for metric_name in metric_names:
        metric = metric_name_to_fn[metric_name]
        metric_vals[metric_name]['baseline'] = metric(y, y_mean)
        metric_vals[metric_name]['held-out'] = metric(y, y_hat)

    var_explained = variance_explained(sqd_err(y, y_mean), sqd_err(y, y_hat))

    return metric_vals, var_explained
