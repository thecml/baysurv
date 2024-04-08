# Probabilistic Survival Analysis by Approximate Bayesian Inference of Neural Networks

This repository is the official TensorFlow implementation of "Probabilistic Survival Analysis by Approximate Bayesian Inference of Neural Networks", 2024.

The proposed method is implemented based on [TensorFlow Probability](https://github.com/tensorflow/probability).

Evaluation is done using [SurvivalEval](https://github.com/shi-ang/SurvivalEVAL). Big thanks to the authors.

<b>Preprint: TBA</b>

<p align="left"><img src="https://github.com/thecml/baysurv/blob/main/img/bnn.png" width="60%" height="60%">

In this paper, we study the benefits of modeling uncertainty in deep neural networks for survival analysis with a focus on prediction and calibration performance. For this, we present a Bayesian deep learning framework that consists of three probabilistic network architectures, which we train by optimizing the Cox partial likelihood and combining aleatoric and epistemic uncertainty.


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
