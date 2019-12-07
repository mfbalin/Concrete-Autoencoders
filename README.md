# Concrete Autoencoders

The concrete autoencoder is an end-to-end differentiable method for global feature selection, which efficiently identifies a subset of the most informative features and simultaneously learns a neural network to reconstruct the input data from the selected features. The method can be applied to unsupervised and supervised settings, and is a modification of the standard autoencoder.

For more details, see the accompanying paper: ["Concrete Autoencoders for Differentiable Feature Selection and Reconstruction"](https://arxiv.org/abs/1901.09346), *ICML 2019*, and please use the citation below.

```
@article{abid2019concrete,
  title={Concrete Autoencoders for Differentiable Feature Selection and Reconstruction},
  author={Abid, Abubakar and Balin, Muhammed Fatih and Zou, James},
  journal={arXiv preprint arXiv:1901.09346},
  year={2019}
}
```

## Installation

To install, use `pip install concrete-autoencoder`

## Usage

Here's an example of using Concrete Autoencoders to select the 20 most important features (pixels) across the entire MNIST dataset:

```python
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (len(x_train), -1))
x_test = np.reshape(x_test, (len(x_test), -1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def decoder(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(784)(x)
    return x

selector = ConcreteAutoencoderFeatureSelector(K = 20, output_function = decoder, num_epochs = 800)

selector.fit(x_train, x_train, x_test, x_test)
```

Then, to get the pixels, run this:
```python
selector.get_support(indices = True)
```

Run this code inside a colab notebook: https://colab.research.google.com/drive/11NMLrmToq4bo6WQ_4WX5G4uIjBHyrzXd

## Documentation:

class ConcreteAutoencoderFeatureSelector:

  Constructor takes a number of parameters to initalize the class. They are:
  
    K: the number of features one wants to select
    
    output_function: the decoder function
    
    num_epochs: number of epochs to start training concrete autoencoders
    
    batch_size: the batch size during training
    
    learning_rate: learning rate of the adam optimizer used during training
    
    start_temp: the starting temperature of the concrete select layer
    
    min_temp: the ending temperature of the concrete select layer
    
    tryout_limit: number of times to double the number of epochs and try again in case it doesn't converge
    
    
  fit(X, Y = None): trains the concrete autoencoder
  
    X: the data for which you want to do feature selection
    
    Y: labels, in case labels are given, it will do supervised feature selection, if not, then unsupervised feature selection
  
  
  transform(X): filters X's features after fit has been called
  
    X: the data to be filtered
    
    
  fit_transform(X): calls fit and transform in a sequence
  
    X: the data to do feature selection on and filter
    
    
  get_support(indices = False): if indices is True, returns indices of the features selected, if not, returns a mask
  
    indices: boolean flag to determine whether to return a list of indices or a boolean mask
    
  
  get_params(): returns the underlying keras model for the concrete autoencoder
    
