import numpy as np
import pandas as pd
import os
import numpy as np
from pathlib import Path
import paths as pt
matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import seaborn as sns
from utility.model import map_model_name

plt.rcParams.update({'axes.labelsize': 'small',
                     'axes.titlesize': 'small',
                     'font.size': 14.0})

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def get_y_label(metric_name):
    if "Loss" in metric_name:
        return r'Loss $\mathcal{L}(\theta)$'
    elif "CTD" in metric_name:
        return 'CTD'
    elif "IBS" in metric_name:
        return "IBS"
    else:
        return "INBLL"

def plot_calibration_curves(percentiles, pred_obs, predictions, model_names, dataset_name):
    n_percentiles = len(percentiles.keys())
    fig, axes = plt.subplots(n_percentiles, 2, figsize=(12, 12))
    labels = list()
    for i, (q, pctl) in enumerate(percentiles.items()):
        for model_idx, model_name in enumerate(model_names):
            pred = pred_obs[pctl][model_name][0]
            obs = pred_obs[pctl][model_name][1]
            preds = predictions[pctl][model_name]
            data = pd.DataFrame({'Pred': pred, 'Obs': obs})
            axes[i][0].set_xlabel("Predicted probability")
            axes[i][1].set_xlabel("Predicted probability")
            axes[i][0].set_ylabel("Observed probability")
            axes[i][0].set_title(f"Calibration at the {q}th percentile of survival time")
            axes[i][1].set_title(f"Probabilities at the {q}th percentile")
            axes[i][0].grid(True)
            axes[i][1].grid(True)
            sns.lineplot(data, x='Pred', y='Obs', color=TFColor[model_idx], ax=axes[i][0], legend=False, label=map_model_name(model_name))
            sns.kdeplot(preds, fill=True, common_norm=True, alpha=.5, cut=0, linewidth=1, color=TFColor[model_idx], ax=axes[i][1])
        ax=axes[i][0].plot([0, 1], [0, 1], c="k", ls="--", linewidth=1.5)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(Path.joinpath(pt.RESULTS_DIR, f"{dataset_name.lower()}_calibration.pdf"),
                format='pdf', bbox_inches="tight")
    plt.close()

def plot_training_curves(results, dataset_name, model_names, metric_names):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for (j, metric_name) in enumerate(metric_names):
        for (k, model_name) in enumerate(model_names):
            model_results = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            metric_results = model_results[metric_name]
            n_epochs = len(model_results)
            axes[j].plot(range(n_epochs), metric_results, label=map_model_name(model_name),
                            marker="o", color=TFColor[k], linewidth=1)
        axes[j].set_xlabel('Epoch', fontsize="medium")
        axes[j].set_ylabel(metric_name, fontsize="medium")
        axes[j].grid()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    plt.savefig(Path.joinpath(pt.RESULTS_DIR, f"{dataset_name.lower()}_training_curves.pdf"),
                format='pdf', bbox_inches="tight")
    plt.close()