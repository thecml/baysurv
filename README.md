# Efficient Training of Probabilistic Neural Networks for Survival Analysis

This repository is the official TensorFlow implementation of "Efficient Training of Probabilistic Neural Networks for Survival Analysis", 2024.

The proposed method is implemented based on [TensorFlow Probability](https://github.com/tensorflow/probability).

Evaluation is done using [SurvivalEval](https://github.com/shi-ang/SurvivalEVAL). Thank you to the authors.

<b>Preprint: TBA</b>

<p align="left"><img src="https://github.com/thecml/baysurv/blob/main/img/bnn.png" width="60%" height="60%">

In the context of survival analysis using Bayesian modeling, we investigate whether non-VI techniques can offer comparable or possibly improved predictive performance and uncertainty calibration compared to VI.

License
--------
To view the license for this work, visit https://github.com/thecml/baysurv/blob/main/LICENSE


Requirements
----------------------
To run the models, please refer to [Requirements.txt](https://github.com/thecml/baysurv/blob/main/requirements.txt).

Code was tested in virtual environment with `Python 3.9`, `TensorFlow 2.11` and `TensorFlow Probability 0.19`


Training
--------
- Make directories `mkdir results` and `mkdir models`.

- Please refer to `paths.py` to set appropriate paths. By default, results are in `results` and models in `models`

- Network configuration using best hyperparameters are found in `configs/*`

- Run the training code:

```
# SOTA models
python train_sota_models.py

# BNN Models
python train_bnn_models.py
```


Evaluation
--------
- After model training, view the results in the `results` folder.


Visualization
---------
- Run the notebook to plot the survival function and the predicted time to event:
```
jupyter notebook model_inference.ipynb
```
