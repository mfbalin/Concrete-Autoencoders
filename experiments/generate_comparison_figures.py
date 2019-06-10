import matplotlib
matplotlib.use('Agg')
figure_dir = 'figures'

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from PIL import Image

import random
import datetime
import os

from concrete_estimator import run_experiment

def concrete_column_subset_selector_general(train, test, K, model_dir = None):
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    ttrain = train
    train = (x_train, y_train)
    val = (x_val, y_val)
    
    probabilities = run_experiment('', train, val, test, K, [K * 3 // 2], 300, max(train[0].shape[0] // 256, 16), 0.001, 0.1)
    indices = np.argmax(probabilities, axis = 1)
    
    return ttrain[0][:, indices], test[0][:, indices]

eps = 1e-12

def load_mice(one_hot = False):
    filling_value = -100000

    X = np.genfromtxt('datasets/Data_Cortex_Nuclear.csv', delimiter = ',', skip_header = 1, usecols = range(1, 78), filling_values = filling_value, encoding = 'UTF-8')
    classes = np.genfromtxt('datasets/Data_Cortex_Nuclear.csv', delimiter = ',', skip_header = 1, usecols = range(78, 81), dtype = None, encoding = 'UTF-8')

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean([X[k, j] for k in range(classes.shape[0]) if np.all(classes[i] == classes[k])])

    DY = np.zeros((classes.shape[0]), dtype = np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ['Control', 'Memantine', 'C/S'])):
            DY[i] += (2 ** j) * (val == label)

    Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
    for idx, val in enumerate(DY):
        Y[idx, val] = 1

    X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    DY = DY[indices]
    classes = classes[indices]
    
    if not one_hot:
        Y = DY
        
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    print(X.shape, Y.shape)
    
    return (X[: X.shape[0] * 4 // 5], Y[: X.shape[0] * 4 // 5]), (X[X.shape[0] * 4 // 5:], Y[X.shape[0] * 4 // 5: ])

import numpy as np

def load_isolet():
    x_train = np.genfromtxt('datasets/isolet1+2+3+4.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
    y_train = np.genfromtxt('datasets/isolet1+2+3+4.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')
    x_test = np.genfromtxt('datasets/isolet5.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
    y_test = np.genfromtxt('datasets/isolet5.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')
    
    X = MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((x_train, x_test)))
    x_train = X[: len(y_train)]
    x_test = X[len(y_train):]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    return (x_train, y_train), (x_test, y_test)

import numpy as np
def load_epileptic():
    filling_value = -100000
    
    X = np.genfromtxt('datasets/data.csv', delimiter = ',', skip_header = 1, usecols = range(1, 179), filling_values = filling_value, encoding = 'UTF-8')
    Y = np.genfromtxt('datasets/data.csv', delimiter = ',', skip_header = 1, usecols = range(179, 180), encoding = 'UTF-8')
    
    X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    print(X.shape, Y.shape)
    
    return (X[: 8000], Y[: 8000]), (X[8000: ], Y[8000: ])

import tensorflow as tf
import numpy as np

def load_data(fashion = False, digit = None, one_hot = False, normalize = False):
    if fashion:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if digit is not None and 0 <= digit and digit <= 9:
        train = test = {y: [] for y in range(10)}
        for x, y in zip(x_train, y_train):
            train[y].append(x)
        for x, y in zip(x_test, y_test):
            test[y].append(x)

        for y in range(10):
            train[y] = np.asarray(train[y])
            test[y] = np.asarray(test[y])

        x_train = train[digit]
        x_test = test[digit]
    
    x_train = x_train.reshape((-1, x_train.shape[1] * x_train.shape[2])).astype(np.float32)
    x_test = x_test.reshape((-1, x_test.shape[1] * x_test.shape[2])).astype(np.float32)

    if one_hot:
        y_train_t = np.zeros((y_train.shape[0], 10))
        y_train_t[np.arange(y_train.shape[0]), y_train] = 1
        y_train = y_train_t
        y_test_t = np.zeros((y_test.shape[0], 10))
        y_test_t[np.arange(y_test.shape[0]), y_test] = 1
        y_test = y_test_t
    
    if normalize:
        X = np.concatenate((x_train, x_test))
        X = (X - X.min()) / (X.max() - X.min())
        x_train = X[: len(y_train)]
        x_test = X[len(y_train):]
    
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)

def load_fashion():
    train, test = load_data(fashion = True, normalize = True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size = 0.6)
    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    train, test = load_data(fashion = False, normalize = True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size = 0.6)
    return (x_train, y_train), (x_test, y_test)

def load_coil():
    samples = []
    for i in range(1, 21):
        for image_index in range(72):
            obj_img = Image.open(os.path.join('datasets/coil-20-proc', 'obj%d__%d.png' % (i, image_index)))
            rescaled = obj_img.resize((20,20))
            pixels_values = [float(x) for x in list(rescaled.getdata())]
            sample = np.array(pixels_values + [i])
            samples.append(sample)
    samples = np.array(samples)
    np.random.shuffle(samples)
    data = samples[:, :-1]
    targets = (samples[:, -1] + 0.5).astype(np.int64)
    data = (data - data.min()) / (data.max() - data.min())
    
    l = data.shape[0] * 4 // 5
    train = (data[:l], targets[:l])
    test = (data[l:], targets[l:])
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)
    return train, test

def load_activity():
    x_train = np.loadtxt(os.path.join('datasets/dataset_uci', 'final_X_train.txt'), delimiter = ',', encoding = 'UTF-8')
    x_test = np.loadtxt(os.path.join('datasets/dataset_uci', 'final_X_test.txt'), delimiter = ',', encoding = 'UTF-8')
    y_train = np.loadtxt(os.path.join('datasets/dataset_uci', 'final_y_train.txt'), delimiter = ',', encoding = 'UTF-8')
    y_test = np.loadtxt(os.path.join('datasets/dataset_uci', 'final_y_test.txt'), delimiter = ',', encoding = 'UTF-8')
    
    X = MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((x_train, x_test)))
    x_train = X[: len(y_train)]
    x_test = X[len(y_train):]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from skfeature.utility import unsupervised_evaluation
from sklearn.neighbors import KNeighborsClassifier
def eval_subset(train, test):
    n_clusters = len(np.unique(train[2]))
    
    clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1)
    clf.fit(train[0], train[2])
    DTacc = float(clf.score(test[0], test[2]))
    
    clf = KNeighborsClassifier(n_neighbors = 1, algorithm = 'brute', n_jobs = 1)
    clf.fit(train[0], train[2])
    acc = float(clf.score(test[0], test[2]))
    
    LR = LinearRegression(n_jobs = -1)
    LR.fit(train[0], train[1])
    MSELR = float(((LR.predict(test[0]) - test[1]) ** 2).mean())
    
    MSE = float((((decoder((train[0], train[1]), (test[0], test[1])) - test[1]) ** 2).mean()))
    
    max_iters = 10
    cnmi, cacc = 0.0, 0.0
    for iter in range(max_iters):
        nmi, acc = unsupervised_evaluation.evaluation(train[0], n_clusters = n_clusters, y = train[2])
        cnmi += nmi / max_iters
        cacc += acc / max_iters
    print('nmi = {:.3f}, acc = {:.3f}'.format(cnmi, cacc))
    print('acc = {:.3f}, DTacc = {:.3f}, MSELR = {:.3f}, MSE = {:.3f}'.format(acc, DTacc, MSELR, MSE))
    return MSELR, MSE, acc, DTacc, float(cnmi), float(cacc)

import numpy as np
from scipy.linalg import qr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def column_subset_selector(A, k):
  eps = 1e-6
  A_scaled = A / np.sqrt(np.sum(np.square(A), axis=0) / (A.shape[0] - 1))
  u, d, v = np.linalg.svd(A_scaled)
  u_, d_, v_ = np.linalg.svd(A, k)
  n = np.where(d_ < eps)[0]
  if(len(n)>0 and k > n[0]):
    k = n[0] - 1
    print("k was reduced to match the rank of A")
  Q, R, P = qr((v[:,:k]).T, pivoting=True)
  indices = P[:k]
  return indices

def pfa_selector(A, k, debug = False):
  class PFA(object):
      def __init__(self, n_features, q=0.5):
          self.q = q
          self.n_features = n_features

      def fit(self, X):
          if not self.q:
              self.q = X.shape[1]

          sc = StandardScaler()
          X = sc.fit_transform(X)

          pca = PCA(n_components=self.q).fit(X)
          self.n_components_ = pca.n_components_
          A_q = pca.components_.T

          kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
          clusters = kmeans.predict(A_q)
          cluster_centers = kmeans.cluster_centers_

          self.indices_ = [] 
          for cluster_idx in range(self.n_features):
            indices_in_cluster = np.where(clusters==cluster_idx)[0]
            points_in_cluster = A_q[indices_in_cluster, :]
            centroid = cluster_centers[cluster_idx]
            distances = np.linalg.norm(points_in_cluster - centroid, axis=1)
            optimal_index = indices_in_cluster[np.argmin(distances)]
            self.indices_.append(optimal_index) 
  
  pfa = PFA(n_features = k)
  pfa.fit(A)
  if debug:
    print('Performed PFW with q=', pfa.n_components_)
  column_indices = pfa.indices_
  return column_indices

def pfa_transform(A, B, k, debug = False):
    indices = pfa_selector(A[0], k, debug)
    return A[0][:, indices], B[0][:, indices]

        
def greedy_subset_selector(A, k, debug = False):
  mdl = LinearRegression(n_jobs = -1)
  n, d = A.shape
  submatrix = np.zeros((n, 0))
  indices = list()
  for i in range(k):
    if debug:
      print(i / k)
    scores = list()
    for j in range(d):
      newmatrix = np.concatenate((submatrix, A[:, j:j+1]), axis=1)
      mdl.fit(newmatrix, A)
      scores.append(mdl.score(newmatrix, A))
    best_column = np.argmax(scores)
    indices.append(best_column)
    submatrix = np.concatenate((submatrix, A[:, best_column:best_column+1]), axis=1)
  if debug:
    print(indices)
  return indices


def pca_subset_selector(A, k, debug = False):
  z_scaler = StandardScaler()
  z_data = z_scaler.fit_transform(A)
  mdl = PCA(n_components=1)
  mdl.fit_transform(z_data)
  feature_importances = mdl.components_[0, :]
  if debug:
    print(feature_importances)
  indices = np.argsort(np.abs(feature_importances))[::-1]
  indices = indices[:k]
  return indices


def random_selector(A, k):
  return np.random.permutation(range(A.shape[1]))[:k]

def random_transform(A, B, k, debug = False):
    indices = random_selector(A[0], k)
    return A[0][:, indices], B[0][:, indices]

# Simple function to compute an "accuracy" for the feature selection
def accuracy(num_unique_indices, num_repeats, selected_columns):
  columns_present = 0
  for i in range(num_unique_indices):
    for j in range(num_repeats*i, num_repeats*(i+1)):
      if j in selected_columns:
        columns_present += 1
        break
  return columns_present/num_unique_indices

from sklearn.decomposition import PCA
def pca_extractor(train, test, K):
    pca = PCA(n_components = K)
    pca.fit(train[0])
    tx_train = pca.transform(train[0])
    tx_test = pca.transform(test[0])
    return tx_train, tx_test

from skfeature.function.sparse_learning_based.UDFS import udfs
from skfeature.function.sparse_learning_based.MCFS import mcfs
from skfeature.function.sparse_learning_based.MCFS import feature_ranking as mcfs_ranking
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.function.similarity_based.lap_score import feature_ranking as lap_score_ranking
from skfeature.function.similarity_based.SPEC import spec
from skfeature.function.similarity_based.SPEC import feature_ranking as spec_ranking

from skfeature.utility.sparse_learning import feature_ranking as udfs_ranking

def mse_check(train, test):
    LR = LinearRegression(n_jobs = -1)
    LR.fit(train[0], train[1])
    MSELR = ((LR.predict(test[0]) - test[1]) ** 2).mean()
    return MSELR

def lap_ours(train, test, K):
    scores = lap_score(train[0])
    indices = lap_score_ranking(scores)[: K]
    return train[0][:, indices], test[0][:, indices]

def spec_ours(train, test, K):
    scores = spec(train[0])
    indices = spec_ranking(scores)[: K]
    return train[0][:, indices], test[0][:, indices]

def udfs_ours(train, test, K, debug = True):
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    bindices = []
    bmse = 1e100
    for gamma in [1e-3, 1e-1, 1e1, 1e3]:
        W = udfs(x_train, verbose = debug, gamma = gamma, max_iter = 40)
        indices = udfs_ranking(W)[: K]
        mse = mse_check((train[0][:, indices], train[1]), (x_val[:, indices], y_val))
        if bmse > mse:
            bmse = mse
            bindices = indices
    if debug:
        print(bindices, bmse)
    return train[0][:, bindices], test[0][:, bindices]

def mcfs_ours(train, test, K, debug = True):
    W = mcfs(train[0], n_selected_features = K, verbose = debug)
    bindices = mcfs_ranking(W)[: K]
    if debug:
        print(bindices)
    return train[0][:, bindices], test[0][:, bindices]

import tensorflow as tf

def next_batch(samples, labels, num):
    # Return a total of `num` random samples and labels.
    idx = np.random.choice(len(samples), num)

    return samples[idx], labels[idx]

def standard_single_hidden_layer_autoencoder(X, units, O):
    reg_alpha = 1e-3
    D = X.shape[1]
    weights = tf.get_variable("weights", [D, units])
    biases = tf.get_variable("biases", [units])
    X = tf.nn.leaky_relu(tf.matmul(X, weights) + biases)
    X = tf.layers.dense(X, O, tf.nn.leaky_relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_alpha))
    return X, weights

def aefs_subset_selector(train, K, epoch_num=1000, alpha=0.1):
    D = train[0].shape[1]
    O = train[1].shape[1]
    learning_rate = 0.001
    
    tf.reset_default_graph()
    
    X = tf.placeholder(tf.float32, (None, D))
    TY = tf.placeholder(tf.float32, (None, O))
    Y, weights = standard_single_hidden_layer_autoencoder(X, K, O)
    
    loss = tf.reduce_mean(tf.square(TY - Y)) + alpha * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(weights), axis=1)), axis=0) + tf.losses.get_total_loss()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()
    
    batch_size = 256
    batch_per_epoch = train[0].shape[0] // batch_size
    
    costs = []
    
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    
    with tf.Session(config = session_config) as sess:
        sess.run(init)
        for ep in range(epoch_num):
            cost = 0
            for batch_n in range(batch_per_epoch):
                imgs, yimgs = next_batch(train[0], train[1], batch_size)
                _, c, p = sess.run([train_op, loss, weights], feed_dict = {X: imgs, TY: yimgs})
                cost += c / batch_per_epoch
            costs.append(cost)
            
    return list(np.argmax(np.abs(p), axis=0)), costs

def AEFS(train, test, K, debug = True):
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    bindices = []
    bmse = 1e100
    for alpha in [1e-3, 1e-1, 1e1, 1e3]:
        indices, _ = aefs_subset_selector(train, K)
        mse = mse_check((train[0][:, indices], train[1]), (x_val[:, indices], y_val))
        if bmse > mse:
            bmse = mse
            bindices = indices
    if debug:
        print(bindices, bmse)
    return train[0][:, bindices], test[0][:, bindices]

def decoder(train, test, debug = True, epoch_num = 200, dropout = 0.1):
    
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    train = (x_train, y_train)
    val = (x_val, y_val)
    
    D = train[0].shape[1]
    O = train[1].shape[1]
    
    learning_rate = 0.001
    
    best_val_cost = 1e100
    best_tt = 0
    for size in [D * 4 // 9, D * 2 // 3, D, D * 3 // 2]:
        
        tf.reset_default_graph()

        training = tf.placeholder(tf.bool, ())
        X = tf.placeholder(tf.float32, (None, D))
        TY = tf.placeholder(tf.float32, (None, O))

        net = tf.layers.dense(X, size, tf.nn.leaky_relu)
        net = tf.layers.dropout(net, rate = dropout, training = training)
        Y = tf.layers.dense(net, O)

        loss = tf.losses.mean_squared_error(Y, TY)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        batch_size = max(train[0].shape[0] // 256, 16)
        batch_per_epoch = train[0].shape[0] // batch_size

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        
        val_cost = 0
        tt = np.zeros((0, train[1].shape[1]), np.float32)
        with tf.Session(config = session_config) as sess:
            sess.run(init)
            for ep in range(epoch_num):
                cost = 0
                for batch_n in range(batch_per_epoch):
                    imgs, yimgs = next_batch(train[0], train[1], batch_size)
                    _, c = sess.run([train_op, loss], feed_dict = {X: imgs, TY: yimgs, training: True})
                    cost += c / batch_per_epoch
                if debug and (ep + 1) % 50 == 0:
                    print('Epoch #' + str(ep + 1) + ' loss: ' + str(cost))
            for i in range(0, len(val[0]), batch_size):
                imgs, yimgs = val[0][i: i + batch_size], val[1][i: i + batch_size]
                c = sess.run(loss, feed_dict = {X: imgs, TY: yimgs, training: False})
                val_cost += c * len(imgs) / len(val[0])
            for i in range(0, len(test[0]), batch_size):
                imgs, yimgs = test[0][i: i + batch_size], test[1][i: i + batch_size]
                t = sess.run(Y, feed_dict = {X: imgs, training: False})
                tt = np.concatenate((tt, t))
        if best_val_cost > val_cost:
            best_val_cost = val_cost
            best_tt = tt
    
    return best_tt

def autoencoder(train, test, K, debug = True, epoch_num = 500, dropout = [0.1, 0.1]):
    
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    val = (x_val, y_val)
    
    D = train[0].shape[1]
    O = train[1].shape[1]
    
    learning_rate = 0.001
    
    best_val_cost = 1e100
    best_ttt = []
    for size in [D * 4 // 9, D * 2 // 3, D, D * 3 // 2]:
    
        tf.reset_default_graph()

        training = tf.placeholder(tf.bool, ())
        X = tf.placeholder(tf.float32, (None, D))
        TY = tf.placeholder(tf.float32, (None, O))

        net = tf.layers.dense(X, size, tf.nn.leaky_relu)
        net = tf.layers.dropout(net, rate = dropout[0], training = training)
        extracted_features = tf.layers.dense(net, K)
        net = tf.layers.dense(extracted_features, size, tf.nn.leaky_relu)
        net = tf.layers.dropout(net, rate = dropout[1], training = training)
        Y = tf.layers.dense(net, O)

        loss = tf.losses.mean_squared_error(Y, TY)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        batch_size = max(train[0].shape[0] // 256, 16)
        batch_per_epoch = train[0].shape[0] // batch_size

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        val_cost = 0
        ttt = []
        with tf.Session(config = session_config) as sess:
            sess.run(init)
            for ep in range(epoch_num):
                cost = 0
                for batch_n in range(batch_per_epoch):
                    imgs, yimgs = next_batch(x_train, y_train, batch_size)
                    _, c = sess.run([train_op, loss], feed_dict = {X: imgs, TY: yimgs, training: True})
                    cost += c / batch_per_epoch
                if debug and (ep + 1) % 50 == 0:
                    print('Epoch #' + str(ep + 1) + ' loss: ' + str(cost))
            for i in range(0, len(val[0]), batch_size):
                imgs, yimgs = val[0][i: i + batch_size], val[1][i: i + batch_size]
                c = sess.run(loss, feed_dict = {X: imgs, TY: yimgs, training: False})
                val_cost += c * len(imgs) / len(val[0])
            for data in [train, test]:
                cost = 0
                tt = np.zeros((0, K), np.float32)
                for i in range(0, len(data[0]), batch_size):
                    imgs, yimgs = data[0][i: i + batch_size], data[1][i: i + batch_size]
                    t, c = sess.run([extracted_features, loss], feed_dict = {X: imgs, TY: yimgs, training: False})
                    cost += c * len(imgs) / len(data[0])
                    tt = np.concatenate((tt, t))
                ttt.append(tt)
                print('final loss: ' + str(cost))
        if best_val_cost > val_cost:
            best_val_cost = val_cost
            best_ttt = ttt
        
    return best_ttt[0], best_ttt[1]

import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from os.path import join

def eval_on_dataset(name, train, test, feature_sizes, debug = False):
    n_clusters = len(np.unique(train[1]))

    algorithms = [pca_extractor, autoencoder, lap_ours, AEFS, concrete_column_subset_selector_general, udfs_ours, mcfs_ours, pfa_transform, random_transform]

    #selected_indices = []
    alg_metrics = {}
    for alg in algorithms:
        nmi_results = {}
        acc_results = {}
        mseLR_results = {}
        mse_results = {}
        class_results = {}
        class_results_DT = {}
        #all_indices = {}
        for k in feature_sizes:
            print('k = {}, algorithm = {}'.format(k, alg.__name__))
            #indices = alg(x_train, enc.transform(y_train.reshape((-1, 1))).toarray(), k)
            tx_train, tx_test = alg((train[0], train[0]), (test[0], test[0]), k)
            #all_indices[k] = indices
            mseLR, mse, acc, acc_DT, cnmi, cacc = eval_subset((tx_train, train[0], train[1]), (tx_test, test[0], test[1]))
            mseLR_results[k] = float(mseLR)
            mse_results[k] = float(mse)
            class_results[k] = float(acc)
            class_results_DT[k] = float(acc_DT)
            nmi_results[k] = float(cnmi)
            acc_results[k] = float(cacc)
        #selected_indices.append((alg.__name__, all_indices))
        metrics = {'NMI': nmi_results, 'ACC': acc_results, 'MSELR': mseLR_results, 'MSE': mse_results, 'CLASS': class_results, 'CLASSDT': class_results_DT}
        alg_metrics[alg.__name__] = metrics
    
    with open(join(figure_dir, name), 'w') as f:
        json.dump(alg_metrics, f)
    
    return alg_metrics
    
def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    
    '''
    for i in range(1):
        
        K = 50
        
        train, test = load_coil()
        eval_on_dataset('%d_coil_%d.json' % (i, K), train, test, [K], True)

        train, test = load_epileptic()
        eval_on_dataset('%d_epileptic_%d.json' % (i, K), train, test, [K], True)
        
        train, test = load_isolet()
        eval_on_dataset('%d_isolet_%d.json' % (i, K), train, test, [K], True)
    
        train, test = load_fashion()
        eval_on_dataset('%d_fashion_%d.json' % (i, K), train, test, [K], True)
        
        train, test = load_mnist()
        eval_on_dataset('%d_mnist_%d.json' % (i, K), train, test, [K], True)
        
        train, test = load_activity()
        eval_on_dataset('%d_activity_%d.json' % (i, K), train, test, [K], True)
    '''
    
    for i in range(11, 14):

        #train, test = load_fashion()
        #eval_on_dataset('%d_fashion.json' % i, train, test, [10, 25, 40, 55, 70, 85], True)
        
        train, test = load_isolet()
        print(train.shape, test.shape)        
        #eval_on_dataset('%d_isolet.json' % i, train, test, [10, 25, 40, 55, 70, 85], True)
        
    
if __name__ == '__main__':
    main()
