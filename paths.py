"""
paths.py
====================================
Module to hold paths of files.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
MLP_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mlp')
RSF_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'rsf')
COX_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'cox')
COXNET_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'coxnet')
COXBOOST_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'coxboost')
DSM_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dsm')
DCM_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dcm')
DCPH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'dcph')
BAYCOX_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'baycox')
BAYMTLR_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'baymtlr')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')