import tensorflow as tf
import numpy as np
import paths as pt
from utility.loss import CoxPHLoss, CoxPHLossGaussian

class Trainer:
    def __init__(self, model, model_name, train_dataset, valid_dataset,
                 test_dataset, optimizer, loss_function, num_epochs, early_stop,
                 patience, n_samples_train, n_samples_valid, n_samples_test):
        self.num_epochs = num_epochs
        self.model = model
        self.model_name = model_name

        self.train_ds = train_dataset
        self.valid_ds = valid_dataset
        self.test_ds = test_dataset

        self.optimizer = optimizer
        self.loss_fn = loss_function
        
        self.n_samples_train = n_samples_train
        self.n_samples_valid = n_samples_valid
        self.n_samples_test = n_samples_test

        # Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.valid_loss_metric = tf.keras.metrics.Mean(name="valid_loss")
        self.test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
        self.train_loss, self.valid_loss = list(), list()
        self.train_variance, self.valid_variance = list(), list()
                
        self.early_stop = early_stop
        self.patience = patience
        
        self.best_valid_nll = np.inf
        self.best_ep = -1
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=f"{pt.MODELS_DIR}", max_to_keep=num_epochs)
        
    def train_and_evaluate(self):
        stop_training = False
        for epoch in range(1, self.num_epochs+1):
            if epoch > 0 and self.model_name == "sngp":
                self.model.layers[-1].reset_covariance_matrix() # reset covmat for SNGP
            self.train(epoch)
            if self.valid_ds is not None:
                stop_training = self.validate(epoch)
            if self.test_ds is not None:
                self.test()
            if stop_training:
                self.cleanup()
                break
            self.cleanup()

    def train(self, epoch):
        batch_variances = list()
        runs = self.n_samples_train
        for x, y in self.train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            n_samples = y_event.shape[0]
            with tf.GradientTape() as tape:
                if self.model_name == "mlp":
                    logits = self.model(x, training=True)
                    batch_variances.append(0)
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                elif self.model_name == "sngp":
                    logits, covmat = self.model(x, training=True)
                    batch_variances.append(np.mean(tf.linalg.diag_part(covmat)[:, None]))
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                elif self.model_name == "vi":
                    logits_dist = self.model(x, training=True)
                    logits_cpd = tf.stack([tf.reshape(logits_dist.sample(), n_samples) for _ in range(runs)])
                    batch_variances.append(np.mean(tf.math.reduce_variance(logits_cpd, axis=0, keepdims=True)))
                    logits_mean = tf.expand_dims(tf.reduce_mean(logits_cpd, axis=0), axis=1)
                    cox_loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
                    self.train_loss_metric.update_state(cox_loss)
                    loss = cox_loss + tf.reduce_mean(self.model.losses) # CoxPHLoss + KL-divergence
                elif self.model_name in ["mcd1", "mcd2", "mcd3"]:
                    logits_dist = self.model(x, training=True)
                    logits_cpd = tf.stack([tf.reshape(logits_dist.sample(), n_samples) for _ in range(runs)])
                    batch_variances.append(np.mean(tf.math.reduce_variance(logits_cpd, axis=0, keepdims=True)))
                    logits_mean = tf.expand_dims(tf.reduce_mean(logits_cpd, axis=0), axis=1)
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
                else:
                    raise NotImplementedError()
                self.train_loss_metric.update_state(loss)
            with tf.name_scope("gradients"):
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        epoch_loss = self.train_loss_metric.result()
        self.train_loss.append(float(epoch_loss))
        
        print(f"Epoch loss: {epoch_loss}")
                
        # Track variance
        if len(batch_variances) > 0:
            self.train_variance.append(float(np.mean(batch_variances)))
        
        self.manager.save()

    def validate(self, epoch):
        stop_training = False
        runs = self.n_samples_valid
        batch_variances = list()
        for x, y in self.valid_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            n_samples = y_event.shape[0]
            if self.model_name == "mlp":
                logits = self.model(x, training=False)
                batch_variances.append(0) # zero variance for MLP
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
            elif self.model_name == "sngp":
                logits, covmat = self.model(x, training=False)
                batch_variances.append(np.mean(tf.linalg.diag_part(covmat)[:, None]))
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
            elif self.model_name == "vi":
                logits_dist = self.model(x, training=False)
                logits_cpd = tf.stack([tf.reshape(logits_dist.sample(), n_samples) for _ in range(runs)])
                batch_variances.append(np.mean(tf.math.reduce_variance(logits_cpd, axis=0, keepdims=True)))
                logits_mean = tf.expand_dims(tf.reduce_mean(logits_cpd, axis=0), axis=1)
                cox_loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
                self.train_loss_metric.update_state(cox_loss)
                loss = cox_loss + tf.reduce_mean(self.model.losses) # CoxPHLoss + KL-divergence
            elif self.model_name in ["mcd1", "mcd2", "mcd3"]:
                logits_dist = self.model(x, training=False)
                logits_cpd = tf.stack([tf.reshape(logits_dist.sample(), n_samples) for _ in range(runs)])
                batch_variances.append(np.mean(tf.math.reduce_variance(logits_cpd, axis=0, keepdims=True)))
                logits_mean = tf.expand_dims(tf.reduce_mean(logits_cpd, axis=0), axis=1)
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
            self.valid_loss_metric.update_state(loss)
        epoch_loss = self.valid_loss_metric.result()
        self.valid_loss.append(float(epoch_loss))

        # Track variance
        if len(batch_variances) > 0:
            self.valid_variance.append(float(np.mean(batch_variances)))

        # Early stopping
        if self.early_stop:
            print(f"{self.model_name} - {epoch}/{self.num_epochs} - {epoch_loss} - {self.best_valid_nll}")
            if self.best_valid_nll > epoch_loss:
                self.best_valid_nll = epoch_loss
                self.best_ep = epoch
            if (epoch - self.best_ep) > self.patience:
                print(f"Validation loss converges at {self.best_ep}th epoch.")
                stop_training = True
            else:
                stop_training = False
                
        return stop_training

    def test(self):
        batch_variances = list()
        for x, y in self.test_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            if self.model_name in ["MLP-ALEA", "VI", "VI-EPI", "MCD-EPI", "MCD"]:
                runs = self.n_samples_test
                logits_cpd = np.zeros((runs, len(x)), dtype=np.float32)
                for i in range(0, runs):
                    if self.model_name in ["MLP-ALEA", "VI", "MCD"]:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False).sample(), len(x))
                    else:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False), len(x))
                logits_mean = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
                batch_variances.append(np.mean(tf.math.reduce_variance(logits_cpd, axis=0, keepdims=True)))
                
                #logits = self.model(x, training=False)
                if isinstance(self.loss_fn, CoxPHLoss):
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
                else:
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_cpd)
                self.test_loss_metric.update_state(loss)
            elif self.model_name == "SNGP":
                logits, covmat = self.model(x, training=False)
                batch_variances.append(np.mean(tf.linalg.diag_part(covmat)[:, None]))
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                self.test_loss_metric.update_state(loss)
            else:
                logits = self.model(x, training=False)
                batch_variances.append(0) # zero variance for MLP
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                self.test_loss_metric.update_state(loss)
         
        # Track variance
        if len(batch_variances) > 0:
            self.test_variance.append(float(np.mean(batch_variances)))

        epoch_loss = self.test_loss_metric.result()
        self.test_loss_scores.append(float(epoch_loss))

    def cleanup(self):
        self.train_loss_metric.reset_states()
        self.valid_loss_metric.reset_states()
        self.test_loss_metric.reset_states()
