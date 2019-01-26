#!/usr/bin/env python

import sys
from os.path import join, exists
from os import makedirs
import math
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from sklearn.metrics import mean_squared_error

class ConcreteSelect(tf.keras.layers.Layer):
    
    def __init__(self, num_outputs):
        super(ConcreteSelect, self).__init__()
        self.num_outputs = num_outputs
    
    def build(self, input_shape):
        self.logits = self.add_variable('logits', shape = (self.num_outputs, input_shape[-1].value))
    
    def call(self, inputs, temp, training = None):
        if training is None:
            training = K.learning_phase()
        
        def samples():
            uniform = tf.random_uniform(shape = self.logits.shape, minval = np.finfo(tf.float32.as_numpy_dtype).tiny, maxval = 1.0)
            gumbel = -tf.log(-tf.log(uniform))
            noisy_logits = (self.logits + gumbel) / temp
            samples = tf.nn.softmax(noisy_logits)
            return samples

        def discrete_logits():
            discrete_logits = tf.one_hot(tf.argmax(self.logits, axis = 1), self.logits.shape[1])
            return discrete_logits
        
        selections = tf.cond(training, samples, discrete_logits)
        outputs = tf.einsum('ij,kj->ik', inputs, selections)
        return outputs

class TemperatureAdjusterHook(tf.train.SessionRunHook):
        
    def __init__(self, temp_placeholder, mean_max, start_temp = 4.0, min_temp = 0.1, alpha = 0.99999):
        self._temp_placeholder = temp_placeholder
        self._mean_max = mean_max
        self._temp = start_temp
        self._min_temp = min_temp
        self._alpha = alpha

    def before_run(self, run_context):
        self._temp = max(self._min_temp, self._temp * self._alpha)
        return tf.train.SessionRunArgs(self._mean_max, feed_dict = {self._temp_placeholder: self._temp})
    
    def after_run(self, run_context, run_values):
        if run_values.results > 0.9980:
            run_context.request_stop()

def concrete_model_fn(features, labels, mode, params):
    
    inputs = tf.feature_column.input_layer(features, params['feature_columns'])
    
    concrete_select = ConcreteSelect(params['num_features'])
    
    trainings = [tf.constant(False), tf.constant(True)]
    temp = tf.placeholder(tf.float32, [], name = 'temp_placeholder')
    nets = [concrete_select(inputs, temp, training) for training in trainings]
    
    for units in params['hidden_units']:
        dense = tf.layers.Dense(units, activation = tf.nn.leaky_relu)
        dropout = tf.layers.Dropout(params['dropout'])
        nets = [dense(net) for net in nets]
        nets = [dropout(net, training = training) for net, training in zip(nets, trainings)]
    
    dense = tf.layers.Dense(params['label_dimension'])
    outputs = [dense(net) for net in nets]
    
    logits = concrete_select.logits
    
    probabilities = tf.nn.softmax(logits)
    max_probabilities = tf.math.reduce_max(probabilities, axis = 1)
    tf.summary.histogram('max_probabilities', max_probabilities)
    indices = tf.argmax(logits, axis = 1)
    tf.summary.text('indices', tf.dtypes.as_string(indices))

    mean_max = tf.math.reduce_mean(max_probabilities)
    
    temperature_adjuster_hook = TemperatureAdjusterHook(temp, mean_max, params['start_temp'], params['min_temp'], params['alpha'])
    temperature_getter_hook = tf.train.FeedFnHook(lambda: {temp: temperature_adjuster_hook._temp})
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': outputs[0],
            'probabilities': probabilities
        }
        return tf.estimator.EstimatorSpec(mode, predictions = predictions, prediction_hooks = [temperature_getter_hook])
    
    loss = tf.losses.mean_squared_error(labels, outputs[1])
    
    optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
    
    prediction_mse = tf.metrics.mean_squared_error(labels, outputs[0])
    
    metrics = {
        'prediction_mse': prediction_mse,
    }
    
    for key, value in metrics.items():
        tf.summary.scalar(key, value[1])
    tf.summary.scalar('temp', temp)
    tf.summary.scalar('mean_max', mean_max)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = metrics, evaluation_hooks = [temperature_getter_hook])
    
    # mode == tf.estimator.ModeKeys.TRAIN
    
    return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op, training_hooks = [temperature_adjuster_hook])

def dataset_input_fn(train, batch_size, num_epochs = 1, seed = None):
    assert train[0].shape[0] == train[1].shape[0]

    def generator():
        np.random.seed(seed)
        perm = np.random.permutation(train[0].shape[0])
        for i in range(0, train[0].shape[0], batch_size):
            ids = perm[i: i + batch_size]
            yield train[0][ids], train[1][ids]

    iterator = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32)).repeat(num_epochs).prefetch(1).make_one_shot_iterator().get_next()
    return {'features': iterator[0]}, iterator[1]

def run_experiment(name, train, val, test, K, hidden_units, num_epochs, batch_size, learning_rate, dropout, start_temp = 10.0, min_temp = 0.01, tryout_limit = 5):
    
    steps_per_epoch = (train[0].shape[0] + batch_size - 1) // batch_size
    epochs_per_evaluation = 50
    feature_columns = [tf.feature_column.numeric_column(key = 'features', shape = [train[0].shape[1]])]

    eval_input_fn = lambda: dataset_input_fn(val, batch_size, seed = 1)
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn, steps = None, throttle_secs = 0)
    
    test_input_fn = lambda: dataset_input_fn(test, batch_size, seed = 1)
    
    for i in range(tryout_limit):
        model_dir = ('./concrete_model_dir_' + name + '_' + str(num_epochs) + '_' + str(datetime.now())).replace(' ', '_').replace(':', '.')
        if not exists(model_dir):
            makedirs(model_dir)
            
        train_input_fn = lambda: dataset_input_fn(train, batch_size, -1)
        train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, max_steps = num_epochs * steps_per_epoch)
        
        alpha = np.exp(np.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))
        
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        
        regressor = tf.estimator.Estimator(
            model_fn = concrete_model_fn,
            model_dir = model_dir,
            config = tf.estimator.RunConfig(save_checkpoints_steps = epochs_per_evaluation * steps_per_epoch, save_summary_steps = steps_per_epoch, log_step_count_steps = steps_per_epoch, session_config = session_config),
            params = {
                'feature_columns': feature_columns,
                'num_features': K,
                'hidden_units': hidden_units,
                'start_temp': start_temp,
                'min_temp': min_temp,
                'alpha': alpha,
                'dropout': dropout,
                'label_dimension': train[1].shape[1],
                'learning_rate': learning_rate
            }
        )

        tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)
        
        predictions = list(regressor.predict(test_input_fn, yield_single_examples = False))
        
        probabilities = predictions[0]['probabilities']
        
        indices = np.argmax(probabilities, axis = 1)
        with open(join(model_dir, 'indices.txt'), 'w') as file:
            file.write(str(list(indices)))
        
        np.random.seed(seed = 1)
        perm = np.random.permutation(test[0].shape[0])
        test_mse = mean_squared_error([pred for prediction in predictions for pred in prediction['predictions']], test[1][perm])
        print('test_mse: ' + str(test_mse))
        
        max_probabilities = np.amax(probabilities, axis = 1)
        mean_max = np.mean(max_probabilities)
        if mean_max > 0.99 or i == tryout_limit - 1:
            return probabilities
        
        num_epochs *= 2
