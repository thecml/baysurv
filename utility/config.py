"""
config.py
====================================
Utility config functions.
"""

from pathlib import Path
import yaml
import paths as pt

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_config(file_path: Path, file_name: str) -> dict:
    """
    Loads a YAML config file
    :param file_path: file path to use
    :param file_name: file name ot use
    :return: config file
    """
    with open(Path.joinpath(file_path, file_name), 'r') as stream:
        settings = yaml.safe_load(stream)
    return settings

def load_model_config(dataset_name: str, model_name: str) -> dict:
    if model_name == "cox":
        config = load_config(pt.COX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "coxnet":
        config = load_config(pt.COXNET_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "dsm":
        config = load_config(pt.DSM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "dcph":
        config = load_config(pt.DCPH_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "dcm":
        config = load_config(pt.DCM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "rsf":
        config = load_config(pt.RSF_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "coxboost":
        config = load_config(pt.COXBOOST_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "baycox":
        config = dotdict(load_config(pt.BAYCOX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
    elif model_name == "baymtlr":
        config = dotdict(load_config(pt.BAYMTLR_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
    return config