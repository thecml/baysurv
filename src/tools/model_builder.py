import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utility.loss import CoxPHLoss
from utility.metrics import CindexMetric
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

class MonteCarloDropout(tf.keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

tfd = tfp.distributions
tfb = tfp.bijectors

def normal_sp(params):
    return tfd.Normal(loc=params[:,0:1],
                      scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))

def normal_fs(params):
    return tfd.Normal(loc=params[:,0:1], scale=1)

def make_cox_model():
    return CoxPHSurvivalAnalysis(alpha=0.0001)

def make_rsf_model():
    return RandomSurvivalForest(random_state=0)

def make_coxnet_model():
    return CoxnetSurvivalAnalysis(fit_baseline_model=True)

def make_baseline_model(input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def make_nobay_model(input_shape, output_dim, layers, activation_fn, dropout_rate):
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    params = tf.keras.layers.Dense(output_dim)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp)(params)
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    return model

def make_vi_model(n_train_samples, input_shape, output_dim, layers,
                  activation_fn, dropout_rate, regularization_pen):
    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    bias_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                 bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                 activity_regularizer=tf.keras.regularizers.L2(regularization_pen),
                                                 kernel_divergence_fn=kernel_divergence_fn,
                                                 bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(inputs)
            else:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                activity_regularizer=tf.keras.regularizers.L2(regularization_pen),
                                kernel_divergence_fn=kernel_divergence_fn,
                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(hidden)
            else:       
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    params = tfp.layers.DenseFlipout(output_dim,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp)(params)
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    return model

def make_mc_model(input_shape, output_dim, layers,
                  activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)  
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)
    params = tf.keras.layers.Dense(output_dim)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp)(params)
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    return model