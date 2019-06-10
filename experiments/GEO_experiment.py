import numpy as np
import tensorflow as tf
from os.path import join
from sklearn.model_selection import train_test_split
from concrete_estimator import run_experiment
from decoder import test_GEO

dataset_dir = 'datasets'

def load_GEO():
    train = np.load(join(dataset_dir, 'bgedv2_XY_tr_float32.npy'), mmap_mode = 'r')
    val = np.load(join(dataset_dir, 'bgedv2_XY_va_float32.npy'), mmap_mode = 'r')
    test = np.load(join(dataset_dir, 'bgedv2_XY_te_float32.npy'), mmap_mode = 'r')
    return (train, train), (val, val), (test, test)

def load_MNIST(fashion = False, supervised = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() if fashion else tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, x_train.shape[1] * x_train.shape[2])).astype(np.float32)
    x_test = x_test.reshape((-1, x_test.shape[1] * x_test.shape[2])).astype(np.float32)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1)
    if supervised:
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    else:
        return (x_train, x_train), (x_val, x_val), (x_test, x_test)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    
    #train, val, test = load_MNIST(fashion = False, supervised = False)
    
    train, val, test = load_GEO()
    
    sz = 9000
    
    for i in range(3):
        probabilities = run_experiment('%d_GEO_linear' % i, train, val, test, 943, [], 5000, 256, 0.0003, 0.1)
        indices = np.argmax(probabilities, axis = 1)
        print(indices)
        #for hidden_units in ([], [sz], [sz, sz], [sz, sz, sz]):
        #    test_GEO('%d_GEO_%d_hidden_layers' % (i, len(hidden_units)), indices, hidden_units)
    
    '''
    for j, i in enumerate([900, 850, 800, 750, 700, 650, 600]):
        if j < 2 or j % 2 == 0:
            continue
        probabilities = run_experiment('%d_GEO_%d' % (j, i), train, val, test, i, [], 5000, 256, 0.0003, 0.1)
        indices = np.argmax(probabilities, axis = 1)
        print(indices)
        test_GEO('%d_GEO_%d_genes_selected' % (j, i), indices, [])
    '''
if __name__ == '__main__':
    main()